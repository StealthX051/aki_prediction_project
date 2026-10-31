import sys
from pathlib import Path

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


@pytest.mark.parametrize("with_split", [False, True])
def test_matplotlib_flow_geometry_separates_long_rows_text_and_downward_arrows(with_split):
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation
    from reporting.cohort_flow import CohortStage, OutcomeSplit, _draw_cohort_flow_fallback
    stages = [CohortStage(title=title, count=count, n_patients=patients,
        removed=removed, removal_reason=reason, detail=detail)
        for title, count, patients, removed, reason, detail in [
            ("All source operations", 6388, 6090, None, None, None),
            ("General anesthesia", 6043, 5783, 345, "Did not meet: General anesthesia", None),
            ("Exclude ASA V/VI; retain missing ASA", 6031, 5781, 12, "ASA V/VI excluded", None),
            ("Required waveform availability", 5779, 5557, 252, "Incomplete required waveforms", None),
            ("Shared AKI/ICU eligibility", 3581, 3442, 2198, "Outside shared AKI/ICU eligibility", None),
            ("Final cohort", 40, 40, 3541, "Outside diagnostic subset (not a clinical exclusion)",
             "Diagnostic subset (enriched; not a clinical exclusion); main_shared; general anesthesia; retained repeat operations; diagnostic subset")]]
    title = "Cohort Construction Flow"
    fig = _draw_cohort_flow_fallback(stages, OutcomeSplit("AKI", 10, 30) if with_split else None, title)
    try:
        fig.canvas.draw()
        ax, renderer = fig.axes[0], fig.canvas.get_renderer()
        patches = [patch.get_window_extent(renderer) for patch in ax.patches]
        labels = [text for text in ax.texts if text.get_text() and text.get_text() != title]
        assert len(labels) == len(patches)
        for index, box in enumerate(patches):
            assert all(not box.overlaps(other) for other in patches[index+1:])
            text = labels[index].get_window_extent(renderer)
            assert box.x0 <= text.x0 < text.x1 <= box.x1
            assert box.y0 <= text.y0 < text.y1 <= box.y1
        arrows = [text for text in ax.texts if isinstance(text, Annotation)]
        assert len(arrows) == len(stages)-1 + (2 if with_split else 0)
        assert all(arrow.get_position()[1] > arrow.xy[1] for arrow in arrows)
    finally:
        plt.close(fig)


def test_dataset_cli_persists_exact_source_and_diagnostic_flow_metadata(tmp_path, monkeypatch):
    import json
    import pandas as pd
    import reporting.cohort_flow as module
    from data_preparation.artifact_metadata import metadata_path_for
    dataset = tmp_path / "cohort.csv"
    pd.DataFrame({"caseid": range(40), "subjectid": range(40),
        "eligible_any_aki": 1, "aki_label": 0,
        "eligible_icu_admission": 1, "y_icu_admit": 0}).to_csv(dataset, index=False)
    source = {"outer_flow_counts": {"source_file": "/original/clinical_data.csv", "stages": [
        {"key": key, "title": title, "count": count, "n_patients": patients}
        for key, title, count, patients in [
            ("source", "All source operations", 6388, 6090),
            ("general_anesthesia", "General anesthesia", 6043, 5783),
            ("asa", "Exclude ASA V/VI; retain missing ASA", 6031, 5781),
            ("waveforms", "Required waveform availability", 5779, 5557)]]},
        "cohort_scopes": {"main_shared": {"operations": 3581, "patients": 3442}},
        "diagnostic_trim": {"diagnostic": True, "cohort_scope": "main_shared",
                            "after_operations": 40, "after_patients": 40}}
    metadata_path_for(dataset).write_text(json.dumps(source))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr(module, "METADATA_DIR", tmp_path / "paper/metadata")
    drawn = []
    monkeypatch.setattr(module, "render_cohort_flow", lambda flow, **kwargs: drawn.append(flow))
    monkeypatch.setattr(sys, "argv", ["cohort_flow", "--dataset", str(dataset), "--cohort-scope", "main_shared"])
    module.main()
    destination = tmp_path / "paper/metadata/cohort_flow_counts.json"
    saved = json.loads(destination.read_text())
    assert saved["cohort_scope"] == "main_shared"
    assert saved["stages"][0]["count"] == 6388
    assert saved["stages"][0]["n_patients"] == 6090
    assert saved["stages"][-1]["count"] == saved["stages"][-1]["n_patients"] == 40
    assert saved["diagnostic_trim"] == source["diagnostic_trim"]
    assert saved["outer_flow_counts"] == source["outer_flow_counts"]
    assert module.normalize_counts(saved) == drawn[0]
    assert json.loads(metadata_path_for(dataset).read_text()) == source
    assert not list(destination.parent.glob("*.tmp"))


def test_branch_flow_metadata_does_not_overwrite_main_counts(tmp_path, monkeypatch):
    from reporting.cohort_flow import CohortFlowData, CohortStage, _persist_flow_metadata
    monkeypatch.setenv('AKI_STORAGE_POLICY', 'off')
    dataset = tmp_path / 'cohort.csv'
    dataset.write_text('caseid,subjectid\n1,10\n2,20\n')
    common = dict(metadata_dir=tmp_path / 'metadata', dataset=dataset,
                  scope='main_shared', branch_roster=None, source_metadata={})
    main = _persist_flow_metadata(CohortFlowData([CohortStage('Final', 2, n_patients=2)]), **common)
    before = main.read_bytes()
    windowed = _persist_flow_metadata(CohortFlowData([CohortStage('Final', 1, n_patients=1)]),
                                     output_name='cohort_flow_windowed', **common)
    assert main.name == 'cohort_flow_counts.json'
    assert windowed.name == 'cohort_flow_windowed_counts.json'
    assert main.read_bytes() == before
