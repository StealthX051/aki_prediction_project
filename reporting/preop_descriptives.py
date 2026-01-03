"""Generate descriptive statistics for the preoperative cohort.

This module loads the finalized preoperative dataset alongside the feature
lists defined in ``data_preparation.step_03_preop_prep`` to generate a
publication-ready table summarizing baseline characteristics. Continuous
features are tested for normality using the Shapiro–Wilk test to decide
between mean/standard deviation or median/interquartile range summaries.
Categorical features are summarized with counts and percentages. Outputs are
saved to HTML and LaTeX under ``results/tables`` with human-readable labels
from :func:`reporting.display_dictionary.DisplayDictionary.feature_label`.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import pandas as pd
from scipy.stats import shapiro

from data_preparation.inputs import PREOP_PROCESSED_FILE
from data_preparation.step_03_preop_prep import CATEGORICAL_COLS, CONTINUOUS_COLS
from reporting.display_dictionary import DisplayDictionary, load_display_dictionary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))
TABLES_DIR = RESULTS_DIR / "tables"


def _load_preop_data(path: Path) -> pd.DataFrame:
    """Load the preoperative dataset from CSV.

    Raises a clear error if the file does not exist or cannot be parsed.
    """

    if not path.exists():
        raise FileNotFoundError(f"Preoperative dataset not found at {path}")

    logger.info("Loading preoperative data from %s", path)
    return pd.read_csv(path)


def _shapiro_p_value(values: pd.Series, max_sample: int, random_state: int) -> Optional[float]:
    """Return the Shapiro–Wilk p-value for the provided series.

    SciPy's Shapiro implementation caps the supported sample size at 5,000 and
    requires at least three observations. Values beyond ``max_sample`` are
    randomly subsampled for efficiency and validity.
    """

    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        return None

    sample = clean
    if len(clean) > max_sample:
        sample = clean.sample(max_sample, random_state=random_state)

    try:
        _, p_value = shapiro(sample)
    except ValueError as exc:
        logger.warning("Shapiro–Wilk test failed: %s", exc)
        return None

    return float(p_value)


def _summarize_continuous(
    series: pd.Series,
    *,
    alpha: float,
    max_sample: int,
    random_state: int,
) -> Tuple[str, Optional[float]]:
    """Summarize a continuous series with normality-guided metrics."""

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "No data", None

    p_value = _shapiro_p_value(numeric, max_sample=max_sample, random_state=random_state)
    if p_value is not None and p_value >= alpha:
        mean = numeric.mean()
        std = numeric.std(ddof=1)
        summary = f"{mean:.2f} ± {std:.2f}"
    else:
        median = numeric.median()
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        summary = f"{median:.2f} ({q1:.2f}, {q3:.2f})"

    return summary, p_value


def _format_category_counts(series: pd.Series) -> str:
    """Return a compact counts/percentages string for a categorical column."""

    if series.empty:
        return "No data"

    filled = series.fillna("Missing")
    total = len(filled)
    counts = filled.value_counts(dropna=False)

    parts: List[str] = []
    for category, count in counts.items():
        proportion = (count / total) * 100
        parts.append(f"{category}: {count} ({proportion:.1f}%)")

    return "; ".join(parts)


def _build_descriptive_table(
    df: pd.DataFrame,
    *,
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    display_dictionary: DisplayDictionary,
    alpha: float,
    max_sample: int,
    random_state: int,
) -> pd.DataFrame:
    """Compile descriptive statistics into a tidy DataFrame."""

    rows: List[Tuple[str, str, str]] = []

    for feature in continuous_features:
        if feature not in df.columns:
            logger.warning("Continuous feature %s not found; skipping", feature)
            continue

        summary, _ = _summarize_continuous(
            df[feature], alpha=alpha, max_sample=max_sample, random_state=random_state
        )
        label = display_dictionary.feature_label(feature, use_short=True, include_unit=True)
        rows.append((label, "Continuous", summary))

    for feature in categorical_features:
        if feature not in df.columns:
            logger.warning("Categorical feature %s not found; skipping", feature)
            continue

        summary = _format_category_counts(df[feature])
        label = display_dictionary.feature_label(feature, use_short=True, include_unit=True)
        rows.append((label, "Categorical", summary))

    return pd.DataFrame(rows, columns=["Feature", "Type", "Summary"])


def _save_table(table: pd.DataFrame, output_prefix: str) -> Tuple[Path, Path]:
    """Persist the table to HTML and LaTeX formats."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    html_path = TABLES_DIR / f"{output_prefix}.html"
    latex_path = TABLES_DIR / f"{output_prefix}.tex"

    table.to_html(html_path, index=False, escape=False)
    table.to_latex(latex_path, index=False, escape=False, longtable=True)

    logger.info("Saved descriptive table to %s and %s", html_path, latex_path)
    return html_path, latex_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PREOP_PROCESSED_FILE,
        help="Path to the finalized preoperative dataset (CSV).",
    )
    parser.add_argument(
        "--display-dictionary",
        type=Path,
        default=None,
        help="Optional path to a custom display dictionary JSON file.",
    )
    parser.add_argument(
        "--output-prefix",
        default="preop_descriptives",
        help="Filename prefix for saved tables under results/tables.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the Shapiro–Wilk normality test.",
    )
    parser.add_argument(
        "--max-normality-sample",
        type=int,
        default=5000,
        help="Maximum sample size to use for the Shapiro–Wilk test.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when subsampling for the Shapiro–Wilk test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = _load_preop_data(args.dataset)
    display_dict = load_display_dictionary(args.display_dictionary)

    descriptive_table = _build_descriptive_table(
        df,
        continuous_features=CONTINUOUS_COLS,
        categorical_features=CATEGORICAL_COLS,
        display_dictionary=display_dict,
        alpha=args.alpha,
        max_sample=args.max_normality_sample,
        random_state=args.random_state,
    )

    if descriptive_table.empty:
        raise ValueError("No descriptive statistics generated; verify dataset columns match expectations.")

    _save_table(descriptive_table, args.output_prefix)


if __name__ == "__main__":
    main()
