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

    flow = normalize_counts(counts, display_dict)

    waveform_stage = next(stage for stage in flow.stages if "Waveform" in stage.title)
    assert waveform_stage.detail is not None
    assert "Pleth" in waveform_stage.detail or "Plethysmography" in waveform_stage.detail
    assert waveform_stage.removed == 1
    assert waveform_stage.removal_reason == "Incomplete waveform data"
    assert flow.stages[0].count == 10
    assert flow.stages[-1].title == "Final cohort"


def test_render_creates_files(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr("reporting.cohort_flow.refresh_paper_bundle", lambda _paper_dir: {})
    monkeypatch.setattr("reporting.cohort_flow.shutil.which", lambda _name: "/usr/bin/dot")
    commands = []

    def _fake_run(command, check):
        assert check is True
        commands.append(command)
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(f"rendered {output_path.suffix}\n", encoding="utf-8")

    monkeypatch.setattr("reporting.cohort_flow.subprocess.run", _fake_run)

    flow = normalize_counts(
        {
            "total_cases": 4,
            "mandatory_columns": {"count": 4, "columns": ["preop_cr", "opend"]},
            "waveforms": {"SNUADC/PLETH": 3},
            "custom_filters": [
                {"name": "filter_preop_cr", "count_before": 3, "count": 2},
            ],
            "final_cohort": 2,
            "label_split": {"label": "AKI", "true": 1, "false": 1},
        }
    )

    output_paths = render_cohort_flow(
        flow,
        output_dir=tmp_path,
        output_name="test_flow",
        formats=("png", "svg"),
        title=None,
    )

    assert all(path.exists() for path in output_paths)
    assert {path.suffix for path in output_paths} == {".dot", ".png", ".svg"}

    dot_text = (tmp_path / "test_flow.dot").read_text(encoding="utf-8")
    assert "High-fidelity" in dot_text
    assert "Baseline Creatinine" in dot_text
    assert "Eligibility" in dot_text
    assert "Baseline creatinine > 4.0" in dot_text
    assert "excluded_" in dot_text
    assert "final_negative" in dot_text
    assert "final_positive" in dot_text
    assert "final_split" not in dot_text
    assert "splines=polyline" in dot_text
    assert ":s -> final_negative:n" in dot_text
    assert ":s -> final_positive:n" in dot_text
    assert any("-Gdpi=600" in command for command in commands)
    assert (tmp_path / "primary_figures" / "test_flow.png").exists()
    assert (tmp_path / "primary_figures" / "test_flow.svg").exists()


def test_render_uses_matplotlib_fallback_when_graphviz_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr("reporting.cohort_flow.refresh_paper_bundle", lambda _paper_dir: {})
    monkeypatch.setattr("reporting.cohort_flow.shutil.which", lambda _name: None)

    flow = normalize_counts(
        {
            "total_cases": 4,
            "waveforms": {"SNUADC/PLETH": 3},
            "final_cohort": 3,
            "label_split": {"label": "AKI", "true": 1, "false": 2},
        }
    )

    output_paths = render_cohort_flow(
        flow,
        output_dir=tmp_path,
        output_name="missing_dot",
        formats=("png", "svg"),
        title=None,
    )

    assert all(path.exists() for path in output_paths)
    assert (tmp_path / "missing_dot.dot").exists()
    assert (tmp_path / "missing_dot.png").exists()
    assert (tmp_path / "missing_dot.svg").exists()
    assert (tmp_path / "primary_figures" / "missing_dot.png").exists()
    assert (tmp_path / "primary_figures" / "missing_dot.svg").exists()


def test_normalize_counts_inserts_pre_waveform_custom_filter_before_waveform_stage():
    counts = {
        "total_cases": 100,
        "mandatory_columns": {"count": 90, "columns": ["preop_cr", "opend"]},
        "waveforms": [
            {"channel": "SNUADC/PLETH", "count": 80},
            {"channel": "SNUADC/ECG_II", "count": 75},
        ],
        "custom_filters": [
            {
                "name": "Exclude ASA V/VI",
                "label": "Excluded ASA V/VI",
                "count_before": 90,
                "count": 88,
            },
            {"name": "filter_preop_cr", "count_before": 75, "count": 70},
        ],
        "final_cohort": {"count": 70},
    }

    flow = normalize_counts(counts)
    titles = [stage.title for stage in flow.stages]

    assert "ASA Class Exclusion" in titles
    assert "High-fidelity Waveform Availability" in titles
    assert titles.index("ASA Class Exclusion") < titles.index("High-fidelity Waveform Availability")

    asa_stage = next(stage for stage in flow.stages if stage.title == "ASA Class Exclusion")
    waveform_stage = next(stage for stage in flow.stages if stage.title == "High-fidelity Waveform Availability")

    assert asa_stage.count == 88
    assert asa_stage.removed == 2
    assert asa_stage.removal_reason == "ASA class V/VI excluded"
    assert waveform_stage.count == 75
    assert waveform_stage.removed == 13
