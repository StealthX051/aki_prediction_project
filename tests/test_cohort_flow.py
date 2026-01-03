import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from reporting.cohort_flow import normalize_counts, render_cohort_flow
from reporting.display_dictionary import load_display_dictionary


@pytest.fixture(scope="module")
def display_dict():
    return load_display_dictionary()


def test_normalize_counts_uses_display_dictionary(display_dict):
    counts = {
        "total_cases": 10,
        "departments": {"count": 8, "departments": ["general surgery"]},
        "mandatory_columns": {"count": 7, "columns": ["preop_cr", "opend"]},
        "waveforms": [
            {"channel": "SNUADC/PLETH", "count": 6},
        ],
        "custom_filters": [
            {"name": "filter_preop_cr", "label": "Baseline creatinine", "count": 5}
        ],
        "final_cohort": {"count": 5},
    }

    stages = normalize_counts(counts, display_dict)

    waveform_stage = next(stage for stage in stages if "Waveform availability" in stage.title)
    assert "Pleth" in waveform_stage.title or "Plethysmography" in waveform_stage.title
    assert stages[0].count == 10
    assert stages[-1].title == "Final cohort"


def test_render_creates_files(tmp_path: Path):
    stages = normalize_counts({
        "total_cases": 4,
        "waveforms": {"SNUADC/PLETH": 3},
        "final_cohort": 3,
    })

    output_paths = render_cohort_flow(
        stages,
        output_dir=tmp_path,
        output_name="test_flow",
        formats=("png",),
        title=None,
    )

    assert output_paths[0].exists()
    assert output_paths[0].suffix == ".png"
