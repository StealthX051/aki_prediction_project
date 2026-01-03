from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting.display_dictionary import DisplayDictionary
from reporting.missingness_table import compute_missingness_table


def _build_display_dictionary() -> DisplayDictionary:
    raw = {
        "features": {
            "feature_a": {"label": "Feature A", "short_label": "Feat A"},
            "feature_b": "Feature B",
        }
    }
    return DisplayDictionary(raw=raw, source=Path("dummy.json"))


def test_compute_missingness_table_excludes_columns_and_sorts():
    df = pd.DataFrame(
        {
            "feature_a": [1, None, 3],
            "feature_b": [None, None, 2],
            "caseid": [10, 11, 12],
        }
    )

    table = compute_missingness_table(
        df,
        _build_display_dictionary(),
        exclude_columns=["caseid"],
    )

    assert list(table["feature_name"]) == ["feature_b", "feature_a"]

    feature_a = table.set_index("feature_name").loc["feature_a"]
    feature_b = table.set_index("feature_name").loc["feature_b"]

    assert feature_b["missing_count"] == 2
    assert feature_a["missing_count"] == 1

    assert feature_b["missing_percent"] == pytest.approx(66.67, rel=1e-3)
    assert feature_a["missing_percent"] == pytest.approx(33.33, rel=1e-3)

    assert feature_a["display_name"] == "Feat A"
    assert feature_b["display_name"] == "Feature B"
