from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting.preop_descriptives import _filter_to_report_cohort


def test_filter_to_report_cohort_defaults_to_aki_modeling_rows():
    df = pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4],
            "eligible_any_aki": pd.Series([1, 1, 0, 1], dtype="Int64"),
            "aki_label": pd.Series([0, 1, None, None], dtype="Int64"),
        }
    )

    filtered = _filter_to_report_cohort(df, "paper_default")

    assert list(filtered["caseid"]) == [1, 2]


def test_filter_to_report_cohort_outer_preserves_all_rows():
    df = pd.DataFrame(
        {
            "caseid": [1, 2, 3],
            "eligible_any_aki": pd.Series([1, 0, 1], dtype="Int64"),
            "aki_label": pd.Series([0, None, 1], dtype="Int64"),
        }
    )

    filtered = _filter_to_report_cohort(df, "outer")

    assert list(filtered["caseid"]) == [1, 2, 3]
