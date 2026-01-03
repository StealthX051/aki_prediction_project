"""Generate a missingness summary table for the merged modeling dataset.

This module loads the saved feature matrix (merged preoperative and intraoperative
data), calculates the proportion of missing values per feature, and saves a table
with both internal column names and publication-ready labels from
``reporting.display_dictionary.DisplayDictionary``. The script is designed to run
independently of model training by consuming the persisted CSV output from
``data_preparation.step_05_data_merge`` (``WIDE_FEATURES_FILE`` by default).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from reporting.display_dictionary import DisplayDictionary, load_display_dictionary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))
TABLES_DIR = RESULTS_DIR / "tables"

DEFAULT_DATASET_PATH = Path(
    os.getenv(
        "MERGED_DATASET_PATH",
        Path(__file__).resolve().parent.parent / "data" / "processed" / "aki_features_master_wide.csv",
    )
)
OUTCOME_COLUMNS = (
    "aki_label",
    "y_severe_aki",
    "y_inhosp_mortality",
    "y_prolonged_los_postop",
    "y_icu_admit",
)
EXCLUDED_COLUMNS = {"caseid", "subject_id", "hadm_id", "split_group", *OUTCOME_COLUMNS}


def _load_dataset(path: Path) -> pd.DataFrame:
    """Read the merged modeling dataset from CSV.

    Raises:
        FileNotFoundError: If the dataset is missing at ``path``.
        pd.errors.ParserError: If the CSV cannot be parsed.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Merged modeling dataset not found at {path}. Provide --dataset to override the default."
        )

    logger.info("Loading merged dataset from %s", path)
    return pd.read_csv(path)


def compute_missingness_table(
    df: pd.DataFrame,
    display_dictionary: DisplayDictionary,
    *,
    exclude_columns: Sequence[str] = (),
    use_short_labels: bool = True,
) -> pd.DataFrame:
    """Return a per-feature missingness summary.

    Args:
        df: Dataset containing features and metadata.
        display_dictionary: Lookup for human-readable feature labels.
        exclude_columns: Columns to ignore when computing missingness.
        use_short_labels: Whether to use short labels when available.
    """

    excluded = set(exclude_columns)
    rows = []
    total_rows = len(df)

    if total_rows == 0:
        raise ValueError("Dataset is empty; cannot compute missingness percentages.")

    for column in df.columns:
        if column in excluded:
            continue

        missing_count = int(df[column].isna().sum())
        missing_percent = (missing_count / total_rows) * 100
        display_name = display_dictionary.feature_label(column, use_short=use_short_labels, include_unit=True)

        rows.append(
            {
                "feature_name": column,
                "display_name": display_name,
                "missing_count": missing_count,
                "missing_percent": round(missing_percent, 2),
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values(by=["missing_percent", "feature_name"], ascending=[False, True]).reset_index(drop=True)


def _save_table(table: pd.DataFrame, output_prefix: str) -> tuple[Path, Path]:
    """Persist the missingness table to CSV and HTML formats."""

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = TABLES_DIR / f"{output_prefix}.csv"
    html_path = TABLES_DIR / f"{output_prefix}.html"

    table.to_csv(csv_path, index=False)
    table.to_html(html_path, index=False, float_format=lambda x: f"{x:.2f}")

    logger.info("Saved missingness table to %s and %s", csv_path, html_path)
    return csv_path, html_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-feature missingness summary table.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=(
            "Path to the merged modeling dataset CSV. Defaults to the WIDE_FEATURES_FILE "
            "from data_preparation inputs or MERGED_DATASET_PATH environment variable."
        ),
    )
    parser.add_argument(
        "--display-dictionary",
        type=Path,
        default=None,
        help="Optional path to a display dictionary JSON file.",
    )
    parser.add_argument(
        "--output-prefix",
        default="missingness_table",
        help="Prefix for output files written to results/tables/.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    display_dictionary = load_display_dictionary(path=args.display_dictionary)
    dataset = _load_dataset(args.dataset)

    table = compute_missingness_table(
        dataset,
        display_dictionary,
        exclude_columns=EXCLUDED_COLUMNS,
        use_short_labels=True,
    )

    _save_table(table, args.output_prefix)


if __name__ == "__main__":
    main()
