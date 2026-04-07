"""Generate report-time XGBoost SHAP figures from saved model bundles."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from artifact_paths import enforce_storage_policy, get_paper_dir, get_results_dir
from model_creation import utils as model_utils
from reporting.display_dictionary import DisplayDictionary, load_display_dictionary
from reporting.figure_rendering import report_figure_style_context, save_report_figure_bundle
from reporting.manuscript_assets import refresh_paper_bundle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = get_results_dir(PROJECT_ROOT)
PAPER_DIR = get_paper_dir(PROJECT_ROOT)
FIGURES_DIR = PAPER_DIR / "figures"
SHAP_SCATTER_DIR = FIGURES_DIR / "shap_scatter"
FEATURED_SCATTER_DIR = FIGURES_DIR / "shap_scatter_featured"

BACKGROUND_SAMPLE_SIZE = 200
MAX_SCATTER_FEATURES = 12
FEATURED_SCATTER_COUNT = 6
MIN_ROWS_PER_CHUNK = 256
NEGATIVE_COLOR = "#2d6ba3"
POSITIVE_COLOR = "#bf3b3b"
TREND_COLOR = "#111827"


@dataclass(frozen=True)
class ShapReportJob:
    outcome: str
    branch: str
    feature_set: str
    prediction_path: Path
    model_dir: Path


@dataclass(frozen=True)
class ShapReportBundle:
    job: ShapReportJob
    predictions: pd.DataFrame
    raw_test_frame: pd.DataFrame
    transformed_test_frame: pd.DataFrame
    shap_values: np.ndarray
    importance: pd.DataFrame


def _safe_name(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_")
    return token or "feature"


def _resolve_parallelism(job_count: int, worker_budget: int) -> tuple[int, int]:
    total_workers = max(1, int(worker_budget))
    total_jobs = max(1, int(job_count))
    if total_jobs == 1:
        return 1, total_workers
    outer_jobs = min(total_jobs, max(1, int(round(total_workers**0.5))))
    per_job_workers = max(1, total_workers // outer_jobs)
    return outer_jobs, per_job_workers


def _prediction_glob(results_dir: Path) -> List[Path]:
    return sorted((results_dir / "models" / "xgboost").glob("*/*/*/predictions/test.csv"))


def _discover_jobs(results_dir: Path) -> List[ShapReportJob]:
    jobs: List[ShapReportJob] = []
    for pred_path in _prediction_glob(results_dir):
        try:
            relative = pred_path.relative_to(results_dir)
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) != 7:
            logger.warning("Skipping unexpected predictions path layout: %s", pred_path)
            continue
        _models, model_type, outcome, branch, feature_set, _predictions, _filename = parts
        if model_type != "xgboost":
            continue
        model_dir = pred_path.parent.parent
        if not (model_dir / "model.json").exists() or not (model_dir / "preprocessor.pkl").exists():
            logger.warning("Skipping %s because model.json or preprocessor.pkl is missing.", model_dir)
            continue
        jobs.append(
            ShapReportJob(
                outcome=outcome,
                branch=branch,
                feature_set=feature_set,
                prediction_path=pred_path,
                model_dir=model_dir,
            )
        )
    return jobs


@lru_cache(maxsize=1)
def _display_dictionary() -> Optional[DisplayDictionary]:
    try:
        return load_display_dictionary()
    except FileNotFoundError:
        logger.warning("Display dictionary not found; SHAP figures will use raw feature names.")
        return None


@lru_cache(maxsize=2)
def _branch_dataset(branch: str) -> pd.DataFrame:
    return model_utils.load_data(branch)


def _load_predictions(path: Path) -> pd.DataFrame:
    prediction_df = pd.read_csv(path).sort_values("caseid").reset_index(drop=True)
    if prediction_df.empty:
        raise ValueError(f"Prediction file is empty: {path}")
    return prediction_df


def _load_preprocessor(path: Path):
    with path.open("rb") as fp:
        return pickle.load(fp)


def _load_xgb_model(path: Path):
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def _align_raw_test_frame(job: ShapReportJob, prediction_df: pd.DataFrame, preprocessor) -> pd.DataFrame:
    branch_df = _branch_dataset(job.branch).copy()
    branch_df["caseid_lookup"] = branch_df["caseid"].astype(str)
    raw_lookup = branch_df.set_index("caseid_lookup")
    ordered_caseids = prediction_df["caseid"].astype(str).tolist()
    missing_caseids = [caseid for caseid in ordered_caseids if caseid not in raw_lookup.index]
    if missing_caseids:
        raise ValueError(
            f"Could not align SHAP rows for {job.prediction_path}; missing caseids in processed dataset: {missing_caseids[:5]}"
        )

    feature_columns = list(preprocessor.numeric_columns_) + list(preprocessor.categorical_columns_)
    missing_columns = sorted(set(feature_columns) - set(branch_df.columns))
    if missing_columns:
        raise ValueError(
            f"Processed dataset for branch '{job.branch}' is missing feature columns required by the saved preprocessor: {missing_columns}"
        )

    raw_test = raw_lookup.loc[ordered_caseids].copy()
    raw_test.index = prediction_df.index
    return raw_test


def _sample_frame(frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= limit:
        return frame.copy()
    return frame.sample(n=limit, random_state=42).copy()


def _compute_shap_values_chunk(model_path: Path, background: pd.DataFrame, explain_chunk: pd.DataFrame) -> np.ndarray:
    import shap

    booster = _load_xgb_model(model_path)
    explainer = shap.TreeExplainer(
        booster,
        data=background,
        feature_perturbation="interventional",
    )
    values = explainer.shap_values(explain_chunk)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    return np.asarray(values)


def _compute_shap_values(
    model_path: Path,
    background: pd.DataFrame,
    explain_frame: pd.DataFrame,
    *,
    worker_budget: int,
) -> np.ndarray:
    if worker_budget <= 1 or len(explain_frame) < 32:
        return _compute_shap_values_chunk(model_path, background, explain_frame)

    chunk_count = min(
        int(worker_budget),
        len(explain_frame),
        max(1, int(len(explain_frame) // MIN_ROWS_PER_CHUNK)),
    )
    if chunk_count <= 1:
        return _compute_shap_values_chunk(model_path, background, explain_frame)

    row_chunks = [chunk for chunk in np.array_split(np.arange(len(explain_frame)), chunk_count) if len(chunk) > 0]
    values_nested = Parallel(n_jobs=len(row_chunks), backend="threading")(
        delayed(_compute_shap_values_chunk)(
            model_path,
            background,
            explain_frame.iloc[chunk].copy(),
        )
        for chunk in row_chunks
    )
    return np.vstack(values_nested)


def _build_importance_table(feature_names: Sequence[str], shap_values: np.ndarray) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    )
    return importance.sort_values(["mean_abs_shap", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _bundle_for_job(job: ShapReportJob, *, shap_calc_workers: int) -> ShapReportBundle:
    prediction_df = _load_predictions(job.prediction_path)
    preprocessor = _load_preprocessor(job.model_dir / "preprocessor.pkl")
    raw_test = _align_raw_test_frame(job, prediction_df, preprocessor)
    transformed_test = preprocessor.transform(raw_test)

    background = _sample_frame(transformed_test, limit=BACKGROUND_SAMPLE_SIZE)
    if background.empty or transformed_test.empty:
        raise ValueError(f"No rows available to explain for {job.prediction_path}")

    shap_values = _compute_shap_values(
        job.model_dir / "model.json",
        background,
        transformed_test,
        worker_budget=shap_calc_workers,
    )
    importance = _build_importance_table(transformed_test.columns.tolist(), shap_values)
    return ShapReportBundle(
        job=job,
        predictions=prediction_df,
        raw_test_frame=raw_test,
        transformed_test_frame=transformed_test,
        shap_values=shap_values,
        importance=importance,
    )


def _feature_label(feature_name: str, *, include_unit: bool = False) -> str:
    display_dictionary = _display_dictionary()
    if display_dictionary is None:
        return feature_name
    return display_dictionary.feature_label(feature_name, include_unit=include_unit)


def _render_empty_plot(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#52606d")


def _style_axes(ax, *, draw_zero_line: bool = False) -> None:
    if draw_zero_line:
        ax.axhline(0.0, color="#7d8793", linestyle="--", linewidth=1.0, alpha=0.85, zorder=1)
    ax.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.16, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#52606d")
    ax.spines["bottom"].set_color("#52606d")
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.05)


def _trendline(feature_vals: np.ndarray, shap_vals: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    valid_mask = np.isfinite(feature_vals) & np.isfinite(shap_vals)
    if int(valid_mask.sum()) < 3:
        return None
    x_valid = feature_vals[valid_mask]
    y_valid = shap_vals[valid_mask]
    if np.unique(x_valid).size < 2:
        return None
    rank_warning = getattr(getattr(np, "exceptions", None), "RankWarning", RuntimeWarning)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", rank_warning)
            slope, intercept = np.polyfit(x_valid, y_valid, deg=1)
    except rank_warning:
        return None
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    trend_x = np.linspace(float(np.nanmin(x_valid)), float(np.nanmax(x_valid)), 128)
    return trend_x, slope * trend_x + intercept


def _scatter_features(bundle: ShapReportBundle) -> List[str]:
    raw_frame = bundle.raw_test_frame
    candidates: List[str] = []
    for feature_name in bundle.importance["feature"].astype(str):
        if feature_name not in raw_frame.columns:
            continue
        series = pd.to_numeric(raw_frame[feature_name], errors="coerce")
        if series.notna().sum() < 2:
            continue
        candidates.append(feature_name)
        if len(candidates) >= MAX_SCATTER_FEATURES:
            break
    return candidates


def _beeswarm_figure(bundle: ShapReportBundle):
    import shap

    with report_figure_style_context():
        plt.figure(figsize=(10.2, max(6.2, min(12.0, len(bundle.transformed_test_frame.columns) * 0.28))))
        shap.summary_plot(
            bundle.shap_values,
            bundle.transformed_test_frame.rename(columns={col: _feature_label(col) for col in bundle.transformed_test_frame.columns}),
            show=False,
            max_display=20,
        )
        figure = plt.gcf()
        plt.tight_layout()
    return figure


def _scatter_figure(bundle: ShapReportBundle, feature_name: str):
    feature_index = bundle.transformed_test_frame.columns.get_loc(feature_name)
    feature_vals = pd.to_numeric(bundle.raw_test_frame[feature_name], errors="coerce").to_numpy(dtype=float, copy=False)
    shap_vals = bundle.shap_values[:, feature_index]
    y_true = bundle.predictions["y_true"].astype(bool).to_numpy(dtype=bool, copy=False)

    with report_figure_style_context():
        fig, ax = plt.subplots(figsize=(8.2, 5.8))
        negative_mask = (~y_true) & np.isfinite(feature_vals) & np.isfinite(shap_vals)
        positive_mask = y_true & np.isfinite(feature_vals) & np.isfinite(shap_vals)
        if negative_mask.any():
            ax.scatter(
                feature_vals[negative_mask],
                shap_vals[negative_mask],
                s=22,
                alpha=0.68,
                color=NEGATIVE_COLOR,
                edgecolors="none",
                label="AKI Negative",
                zorder=3,
            )
        if positive_mask.any():
            ax.scatter(
                feature_vals[positive_mask],
                shap_vals[positive_mask],
                s=22,
                alpha=0.74,
                color=POSITIVE_COLOR,
                edgecolors="none",
                label="AKI Positive",
                zorder=4,
            )
        trend = _trendline(feature_vals, shap_vals)
        if trend is not None:
            ax.plot(trend[0], trend[1], color=TREND_COLOR, linewidth=2.0, label="Linear Trend", zorder=5)
        elif not negative_mask.any() and not positive_mask.any():
            _render_empty_plot(ax, "No finite raw display values available")

        ax.set_xlabel(_feature_label(feature_name, include_unit=True))
        ax.set_ylabel("SHAP Value")
        ax.set_title(f"XGBoost SHAP Scatter\n{_feature_label(feature_name)}")
        _style_axes(ax, draw_zero_line=True)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
    return fig


def _write_importance_csv(bundle: ShapReportBundle) -> Path:
    destination = FIGURES_DIR / f"shap_importance_{bundle.job.outcome}_{bundle.job.branch}_{bundle.job.feature_set}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle.importance.to_csv(destination, index=False)
    return destination


def _generate_job_outputs(job: ShapReportJob, *, shap_calc_workers: int) -> List[Path]:
    bundle = _bundle_for_job(job, shap_calc_workers=shap_calc_workers)
    outputs: List[Path] = [_write_importance_csv(bundle)]

    beeswarm = _beeswarm_figure(bundle)
    try:
        outputs.extend(
            save_report_figure_bundle(
                beeswarm,
                FIGURES_DIR / f"shap_beeswarm_{job.outcome}_{job.branch}_{job.feature_set}",
                close=False,
                mirror_to_primary=True,
            )
        )
    finally:
        plt.close(beeswarm)

    scatter_features = _scatter_features(bundle)
    featured = set(scatter_features[:FEATURED_SCATTER_COUNT])
    for feature_name in scatter_features:
        figure = _scatter_figure(bundle, feature_name)
        try:
            base_name = f"shap_scatter_{job.outcome}_{job.branch}_{job.feature_set}_{_safe_name(feature_name)}"
            outputs.extend(
                save_report_figure_bundle(
                    figure,
                    SHAP_SCATTER_DIR / base_name,
                    close=False,
                    mirror_to_primary=False,
                )
            )
            if feature_name in featured:
                outputs.extend(
                    save_report_figure_bundle(
                        figure,
                        FEATURED_SCATTER_DIR / base_name,
                        close=False,
                        mirror_to_primary=False,
                    )
                )
        finally:
            plt.close(figure)

    return outputs


def generate_shap_outputs(*, n_jobs: int) -> List[Path]:
    enforce_storage_policy(
        {
            "paper_figures_dir": FIGURES_DIR,
            "paper_primary_figures_dir": FIGURES_DIR / "primary_figures",
            "paper_shap_scatter_dir": SHAP_SCATTER_DIR,
            "paper_shap_scatter_featured_dir": FEATURED_SCATTER_DIR,
        }
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_SCATTER_DIR.mkdir(parents=True, exist_ok=True)
    FEATURED_SCATTER_DIR.mkdir(parents=True, exist_ok=True)

    jobs = _discover_jobs(RESULTS_DIR)
    if not jobs:
        logger.warning("No saved XGBoost prediction jobs were discovered under %s.", RESULTS_DIR)
        return []

    outer_jobs, per_job_workers = _resolve_parallelism(len(jobs), n_jobs)
    logger.info(
        "Generating SHAP figures for %d saved XGBoost jobs with outer_jobs=%d and per_job_workers=%d.",
        len(jobs),
        outer_jobs,
        per_job_workers,
    )
    outputs_nested = Parallel(n_jobs=outer_jobs, backend="loky")(
        delayed(_generate_job_outputs)(job, shap_calc_workers=per_job_workers)
        for job in jobs
    )
    outputs = [path for group in outputs_nested for path in group]
    refresh_paper_bundle(PAPER_DIR)
    return outputs


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, ((os.cpu_count() or 1) - 1)),
        help="Total worker budget for parallel SHAP figure generation.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    outputs = generate_shap_outputs(n_jobs=args.n_jobs)
    if outputs:
        logger.info("Generated %d SHAP report artifacts under %s.", len(outputs), FIGURES_DIR)


if __name__ == "__main__":
    main()
