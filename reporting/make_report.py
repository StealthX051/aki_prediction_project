"""Generate tables and figures from standardized reporting artifacts.

This script consumes `results/tables/metrics_summary.csv` (produced by
`results_recreation.metrics_summary`) and the saved prediction files under
`results/**/predictions/test.csv` to reproduce the styled tables and plots
without recalibrating or re-thresholding models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from results_recreation.metrics_summary import PredictionSet, crawl_predictions
from results_recreation.results_analysis import (
    FIGURES_DIR,
    RESULTS_DIR,
    TABLES_DIR,
    generate_docx_report,
    generate_html_tables,
    generate_pdf_report,
    plot_curves,
    prepare_metrics_for_display,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load the consolidated metrics summary CSV."""

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics summary not found at {metrics_path}. Run results_recreation.metrics_summary first."
        )

    return pd.read_csv(metrics_path)


def _prediction_sets_to_frame(prediction_sets: Iterable[PredictionSet]) -> pd.DataFrame:
    """Combine validated prediction sets into a single DataFrame for plotting."""

    frames = []
    for pred_set in prediction_sets:
        df = pred_set.df.copy()
        df["outcome"] = pred_set.outcome
        df["branch"] = pred_set.branch
        df["feature_set"] = pred_set.feature_set
        df["model_name"] = pred_set.model_name

        # Ensure calibrated column exists for plotting fallbacks
        if "y_prob_calibrated" not in df.columns and "y_prob_raw" in df.columns:
            df["y_prob_calibrated"] = df["y_prob_raw"]

        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    metrics_path = TABLES_DIR / "metrics_summary.csv"
    metrics_df = prepare_metrics_for_display(_load_metrics(metrics_path))

    logger.info("Generating styled tables and reports from %s", metrics_path)
    generate_html_tables(metrics_df)
    generate_docx_report(metrics_df)
    generate_pdf_report(metrics_df)

    logger.info("Loading prediction files for plots")
    prediction_sets = crawl_predictions(RESULTS_DIR)
    prediction_frame = _prediction_sets_to_frame(prediction_sets)
    if prediction_frame.empty:
        logger.warning("No prediction files found; skipping plots.")
    else:
        plot_curves(prediction_frame)
        logger.info("Plots saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
