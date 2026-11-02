"""Generate descriptive statistics for the preoperative cohort.

Source cohort values provide the baseline characteristics; the unfitted Step 03
table supplies derived variables absent from that source. Continuous variables
use median (IQR), and categorical variables use counts and percentages, with
operation/patient denominators and missingness. Outputs are written to the
scoped paper tables directory using the display dictionary's clinical labels.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

from artifact_paths import enforce_storage_policy, get_scoped_results_dir, get_scoped_paper_dir
from data_preparation.inputs import COHORT_FILE, PREOP_PROCESSED_FILE
from data_preparation.outcome_registry import DEFAULT_OUTCOME_SPEC
from data_preparation.cohort_scopes import COHORT_SCOPES, select_cohort, selected_case_hash
from data_preparation.step_03_preop_prep import CATEGORICAL_COLS, CONTINUOUS_COLS
from reporting.display_dictionary import DisplayDictionary, load_display_dictionary
from reporting.manuscript_assets import refresh_paper_bundle, write_markdown_sections, write_pdf_sections

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = get_scoped_results_dir(PROJECT_ROOT, os.getenv("COHORT_SCOPE", "main_shared"), os.getenv("VALIDATION_SCHEME", "nested_cv"))
PAPER_DIR = get_scoped_paper_dir(PROJECT_ROOT, os.getenv("COHORT_SCOPE", "main_shared"), os.getenv("VALIDATION_SCHEME", "nested_cv"))
TABLES_DIR = PAPER_DIR / "tables"
ORDERED_CATEGORICALS: Dict[str, Sequence[str]] = {
    # Preserve clinical severity ordering rather than frequency sorting.
    "asa": ("1.0", "2.0", "3.0", "4.0", "5.0", "6.0"),
}
BINARY_LABELS: Dict[str, Dict[str, str]] = {
    # Map binary codes to human-friendly labels for readability.
    "emop": {"0": "False", "1": "True"},
    "preop_htn": {"0": "False", "1": "True"},
    "preop_dm": {"0": "False", "1": "True"},
}
REPORT_COHORT_CHOICES = ("paper_default", "outer", *COHORT_SCOPES)

SECTION_FEATURES = {
    "Demographics": ("age", "sex", "height", "weight", "bmi"),
    "Surgical characteristics": ("emop", "department", "optype", "approach", "asa", "ane_type", "inpatient_preop"),
    "Clinical comorbidities and tests": ("preop_htn", "preop_dm", "preop_ecg", "preop_pft"),
}
TABLE_FOOTNOTE = ("Continuous variables are median (IQR); categorical variables are n (% of all operations). "
                  "Missing/unknown categories are retained. Counts describe operations; repeat operations are retained. "
                  "Source surgical categories and model-side ECG encoding are unchanged. "
                  "Derived preoperative variables come from the unfitted Step 03 table; source values are not winsorized for Table 1.")


def _load_preop_data(path: Path) -> pd.DataFrame:
    """Load the preoperative dataset from CSV.

    Raises a clear error if the file does not exist or cannot be parsed.
    """

    if not path.exists():
        raise FileNotFoundError(f"Preoperative dataset not found at {path}")

    logger.info("Loading preoperative data from %s", path)
    return pd.read_csv(path)


def _load_optional_processed_data(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Load an optional processed preoperative dataset when available."""

    if path is None:
        return None

    if not path.exists():
        logger.warning("Processed dataset not found at %s; derived variables absent from the source will be omitted.", path)
        return None

    logger.info("Loading processed preoperative data from %s", path)
    return pd.read_csv(path)


def _filter_to_report_cohort(df: pd.DataFrame, report_cohort: str) -> pd.DataFrame:
    if report_cohort in COHORT_SCOPES:
        return select_cohort(df, report_cohort)
    if report_cohort == "outer":
        return df.copy()

    eligibility_col = DEFAULT_OUTCOME_SPEC.eligibility_col
    target_col = DEFAULT_OUTCOME_SPEC.target_col
    required = [eligibility_col, target_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing columns required for report_cohort='{report_cohort}': {missing}"
        )

    return df[df[eligibility_col].eq(1) & df[target_col].isin([0, 1])].copy()


def _summarize_continuous(series: pd.Series, *, alpha=0.05, max_sample=5000, random_state=42):
    """Return median (IQR); legacy normality arguments are ignored."""
    numeric = pd.to_numeric(series, errors="coerce").replace([float("inf"), -float("inf")], pd.NA).dropna()
    if numeric.empty:
        return "No data", None, "median (IQR)"
    q1, median, q3 = numeric.quantile([0.25, 0.5, 0.75])
    return f"{median:.2f} ({q1:.2f}, {q3:.2f})", None, "median (IQR)"


def _ecg_display(series: pd.Series) -> pd.Series:
    """Collapse recorded findings for display without changing model input values."""
    def label(value):
        if pd.isna(value) or value in (float("inf"), -float("inf")) or not str(value).strip():
            return "Missing"
        text = str(value).strip()
        if text.lower() in {"unknown", "not known", "not recorded", "not available", "n/a"}:
            return "Unknown"
        return "Normal sinus rhythm" if text.lower() == "normal sinus rhythm" else "Abnormal"
    return series.map(label)


def _restrict_to_roster(df: pd.DataFrame, roster: Optional[Path]) -> pd.DataFrame:
    if roster is None:
        return df
    frozen = pd.read_csv(roster)
    caseids = frozen["caseid"]
    if caseids.duplicated().any() or df["caseid"].duplicated().any():
        raise ValueError("Duplicate operation IDs in descriptive data or roster")
    selected = df.loc[df["caseid"].isin(caseids)].copy()
    if set(selected["caseid"]) != set(caseids):
        raise ValueError("Descriptive dataset does not cover the frozen branch roster")
    if "subjectid" in frozen and selected_case_hash(selected) != selected_case_hash(frozen):
        raise ValueError("Descriptive operation/patient identities do not match the frozen branch roster")
    return selected


def _format_category_counts(
    series: pd.Series,
    category_order: Optional[Sequence[str]] = None,
    category_labels: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, int, float]]:
    """Return ordered category/count/percent tuples for a categorical column."""

    if series.empty:
        return []

    blank = series.astype("string").str.strip().eq("").fillna(False)
    nonfinite = series.isin([float("inf"), -float("inf")])
    filled = series.astype(object).mask(blank | nonfinite).fillna("Missing")
    total = len(filled)
    counts = filled.value_counts(dropna=False)
    ordered_items: List[Tuple[str, int]] = []

    if category_order:
        desired_order = [str(cat) for cat in category_order]
        counts.index = counts.index.astype(str)

        for label in desired_order:
            if label in counts:
                ordered_items.append((label, int(counts[label])))

        for label, count in counts.items():
            if label not in desired_order:
                ordered_items.append((str(label), int(count)))
    else:
        ordered_items = [(str(label), int(count)) for label, count in counts.items()]

    return [
        (category_labels.get(cat, cat) if category_labels else cat, count, (count / total) * 100)
        for cat, count in ordered_items
    ]


def _filter_and_validate_features(
    df: pd.DataFrame,
    *,
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    dataset_path: Path,
) -> Tuple[List[str], List[str]]:
    """Ensure required columns exist and prevent silent omission of categoricals."""

    missing_continuous = [col for col in continuous_features if col not in df.columns]
    missing_categorical = [col for col in categorical_features if col not in df.columns]

    if missing_continuous:
        logger.warning("Missing continuous features in %s: %s", dataset_path, ", ".join(missing_continuous))

    if missing_categorical:
        message = (
            "Dataset %s is missing categorical features (%s). "
            "Provide --dataset pointing to a dataset that still retains the raw categorical columns "
            "(for example COHORT_FILE or PREOP_PROCESSED_FILE from step_03_preop_prep.py)."
            % (dataset_path, ", ".join(missing_categorical))
        )
        raise ValueError(message)

    return [c for c in continuous_features if c in df.columns], [c for c in categorical_features if c in df.columns]


def _build_descriptive_table(
    df_raw: pd.DataFrame,
    df_processed: Optional[pd.DataFrame],
    *,
    continuous_features: Sequence[str],
    categorical_features: Sequence[str],
    display_dictionary: DisplayDictionary,
    alpha: float,
    max_sample: int,
    random_state: int,
) -> pd.DataFrame:
    """Compile descriptive statistics into a tidy DataFrame."""

    if df_raw.empty:
        raise ValueError("Cannot describe an empty cohort")
    population = f"{len(df_raw):,} operations"
    if "subjectid" in df_raw:
        population += f" among {df_raw['subjectid'].nunique():,} unique patients"
    rows = [("Population", "Population", population, "")]
    features = list(dict.fromkeys([*continuous_features, *categorical_features]))
    assigned = {feature for names in SECTION_FEATURES.values() for feature in names}
    sections = {**SECTION_FEATURES, "Laboratory values": tuple(f for f in features if f not in assigned)}
    for section, names in sections.items():
        present = [feature for feature in names if feature in features]
        if not present:
            continue
        rows.append((section, "Section", "", ""))
        for feature in present:
            source = df_raw[feature]  # Table 1 describes source values, never model-fitted clipping.
            label = display_dictionary.feature_label(feature, use_short=True, include_unit=True)
            blank = source.astype("string").str.strip().eq("").fillna(False)
            nonfinite = source.isin([float("inf"), -float("inf")])
            missing = int((source.isna() | blank | nonfinite).sum())
            if feature in continuous_features:
                numeric = pd.to_numeric(source, errors="coerce")
                missing = int((numeric.isna() | numeric.isin([float("inf"), -float("inf")])).sum())
                summary, _, _ = _summarize_continuous(source)
                rows.append((label, "Continuous", summary, f"{missing} ({100 * missing / len(df_raw):.1f}%)"))
            else:
                display_source = _ecg_display(source) if feature == "preop_ecg" else source
                category_rows = _format_category_counts(display_source,
                    category_order=("Normal sinus rhythm", "Abnormal", "Missing", "Unknown") if feature == "preop_ecg" else ORDERED_CATEGORICALS.get(feature),
                    category_labels=BINARY_LABELS.get(feature))
                rows.append((label, "Categorical", "", f"{missing} ({100 * missing / len(df_raw):.1f}%)"))
                for category, count, pct in category_rows:
                    rows.append((f"  {category}", "", f"{count} ({pct:.1f}%)", ""))
    result = pd.DataFrame(rows, columns=["Feature", "Type", "Summary", "Missing n (%)"])
    result.attrs["footnote"] = f"Population: {population}. " + TABLE_FOOTNOTE
    result.attrs["n_operations"] = len(df_raw)
    result.attrs["n_patients"] = df_raw["subjectid"].nunique() if "subjectid" in df_raw else None
    return result


def _save_table(table: pd.DataFrame, output_prefix: str) -> Tuple[Path, Path, Path]:
    """Persist the table to manuscript-facing formats."""

    enforce_storage_policy({"paper_tables_dir": TABLES_DIR})
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = TABLES_DIR / f"{output_prefix}.csv"
    html_path = TABLES_DIR / f"{output_prefix}.html"
    latex_path = TABLES_DIR / f"{output_prefix}.tex"
    md_path = TABLES_DIR / f"{output_prefix}.md"
    docx_path = TABLES_DIR / f"{output_prefix}.docx"
    pdf_path = TABLES_DIR / f"{output_prefix}.pdf"
    footnote = table.attrs.get("footnote", TABLE_FOOTNOTE)

    table.to_csv(csv_path, index=False)
    table.to_html(html_path, index=False, escape=False)
    with html_path.open("a") as handle:
        handle.write(f"<p>{footnote}</p>")
    table.to_latex(latex_path, index=False, escape=True, longtable=True)
    with latex_path.open("a") as handle:
        handle.write("\n\\par{\\footnotesize " + footnote.replace("%", "\\%") + "}\n")
    _save_docx(table, docx_path)
    write_markdown_sections(md_path, title="Preoperative Descriptive Statistics", sections=[{"heading": output_prefix, "dataframe": table}, {"heading": "Notes", "text": footnote}])
    write_pdf_sections(pdf_path, title="Preoperative Descriptive Statistics", sections=[{"heading": output_prefix, "dataframe": table}, {"heading": "Notes", "text": footnote}])
    refresh_paper_bundle(PAPER_DIR)

    logger.info(
        "Saved descriptive table to %s, %s, %s, %s, %s, and %s",
        csv_path,
        html_path,
        latex_path,
        md_path,
        docx_path,
        pdf_path,
    )
    return html_path, latex_path, docx_path


def _save_docx(table: pd.DataFrame, path: Path) -> None:
    """Write the descriptive table to a DOCX document."""

    document = Document()
    document.add_heading("Preoperative Descriptive Statistics", level=1)

    doc_table = document.add_table(rows=1 + len(table), cols=table.shape[1])
    doc_table.style = "Table Grid"

    header_cells = doc_table.rows[0].cells
    for idx, column in enumerate(table.columns):
        header_cells[idx].text = str(column)

    for row_idx, (_, row) in enumerate(table.iterrows(), start=1):
        cells = doc_table.rows[row_idx].cells
        for col_idx, value in enumerate(row):
            cells[col_idx].text = str(value)

    document.add_paragraph(table.attrs.get("footnote", TABLE_FOOTNOTE)).alignment = WD_ALIGN_PARAGRAPH.LEFT
    document.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=COHORT_FILE,
        help=(
            "Path to the preoperative dataset (CSV) that contains the raw categorical columns. "
            "COHORT_FILE and PREOP_PROCESSED_FILE are both valid inputs."
        ),
    )
    parser.add_argument(
        "--processed-dataset",
        type=Path,
        default=PREOP_PROCESSED_FILE,
        help=(
            "Optional unfitted Step 03 table supplying derived variables absent from --dataset; "
            "existing source values are never replaced."
        ),
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
        help="Filename prefix for saved tables under the scoped paper tables directory.",
    )
    parser.add_argument("--cohort-scope", choices=COHORT_SCOPES, default=os.getenv("COHORT_SCOPE", "main_shared"))
    parser.add_argument("--branch-roster", type=Path, default=None)
    parser.add_argument(
        "--report-cohort",
        choices=REPORT_COHORT_CHOICES,
        default=None,
        help=(
            "Override --cohort-scope. 'paper_default' selects the legacy AKI-eligible cohort; "
            "'outer' retains all supplied source rows."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Deprecated compatibility argument; publication summaries always use median (IQR).",
    )
    parser.add_argument(
        "--max-normality-sample",
        type=int,
        default=5000,
        help="Deprecated compatibility argument.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Deprecated compatibility argument.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_cohort = args.report_cohort or args.cohort_scope
    df_raw = _restrict_to_roster(_filter_to_report_cohort(_load_preop_data(args.dataset), report_cohort), args.branch_roster)
    df_processed = _load_optional_processed_data(args.processed_dataset)
    if df_processed is not None:
        df_processed = _restrict_to_roster(_filter_to_report_cohort(df_processed, report_cohort), args.branch_roster)
    if df_processed is not None:
        if (df_processed["caseid"].duplicated().any() or df_raw["caseid"].duplicated().any()
                or set(df_processed["caseid"]) != set(df_raw["caseid"])):
            raise ValueError("Processed preoperative variables do not match the descriptive operation roster")
        derived = [column for column in [*CONTINUOUS_COLS, "inpatient_preop"]
                   if column not in df_raw and column in df_processed]
        if derived:
            df_raw = df_raw.merge(df_processed[["caseid", *derived]], on="caseid", how="left", validate="one_to_one")

    display_dict = load_display_dictionary(args.display_dictionary)
    total_n = len(df_raw)

    continuous_features, categorical_features = _filter_and_validate_features(
        df_raw,
        continuous_features=CONTINUOUS_COLS,
        categorical_features=[*CATEGORICAL_COLS, *(["inpatient_preop"] if "inpatient_preop" in df_raw else [])],
        dataset_path=args.dataset,
    )

    descriptive_table = _build_descriptive_table(
        df_raw,
        df_processed,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        display_dictionary=display_dict,
        alpha=args.alpha,
        max_sample=args.max_normality_sample,
        random_state=args.random_state,
    )

    descriptive_table.columns = [
        "Characteristic",
        "Type",
        f"Finding (N={total_n})",
        "Missing n (%)",
    ]

    if descriptive_table.empty:
        raise ValueError("No descriptive statistics generated; verify dataset columns match expectations.")

    _save_table(descriptive_table, args.output_prefix)


if __name__ == "__main__":
    main()
