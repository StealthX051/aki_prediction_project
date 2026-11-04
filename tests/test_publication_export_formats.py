"""Publication exports preserve saved values while sharing display conventions."""
from pathlib import Path
import re

from docx import Document
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.path import Path as PlotPath
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from reporting import manuscript_assets as assets
from reporting.figure_rendering import save_report_figure_bundle
from results_recreation import results_analysis as analysis


def _metrics():
    records = []
    for index, feature in enumerate(("preop_only", "all_waveforms")):
        row = {"outcome": "any_aki", "branch": "non_windowed",
               "feature_set": feature, "model_name": "EBM", "n": 100,
               "n_patients": 90, "prevalence": .1, "analysis_status": "definitive"}
        for metric in analysis.METRIC_DISPLAY_MAP:
            point = (.12 - .02 * index) if metric == "brier" else (.4 + .1 * index)
            difference = (-.02 if metric == "brier" else .1) * index
            row.update({metric: point, f"{metric}_lower": point - .01,
                        f"{metric}_upper": point + .01, f"delta_{metric}": difference,
                        f"delta_{metric}_lower": difference - .01,
                        f"delta_{metric}_upper": difference + .01})
        records.append(row)
    return analysis.prepare_metrics_for_display(pd.DataFrame(records))


def test_figure_bundle_has_vector_and_explicit_raster_resolution(tmp_path):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.plot([0, 1], [0, 1])
    saved = save_report_figure_bundle(fig, tmp_path / "figures" / "example",
                                    mirror_to_primary=True)
    assert {path.suffix for path in saved} == {".svg", ".png", ".jpg"}
    for suffix, dpi in (("png", 600), ("jpg", 300)):
        image = Image.open(tmp_path / "figures" / f"example.{suffix}")
        assert image.info["dpi"] == pytest.approx((dpi, dpi), abs=.1)
        if suffix == "jpg":
            assert image.mode == "RGB"
            assert min(image.getpixel((0, 0))) >= 250
        assert (tmp_path / "figures" / "primary_figures" / f"example.{suffix}").exists()
    manifest = assets.refresh_paper_bundle(tmp_path)
    entry = next(item for item in manifest["assets"] if item["id"] == "example")
    assert set(entry["paths"]) == {"svg", "png", "jpg"}
    assert set(entry["alternate_paths"]) == {"svg", "png", "jpg"}
    assert not plt.fignum_exists(fig.number)
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.text(.5, .5, "Vector text")
    variants = assets.save_figure_bundle(fig, tmp_path / "optional", formats=(".PNG", "PDF"))
    assert {path.suffix for path in variants} == {".png", ".pdf"}
    assert b"/FontFile2" in (tmp_path / "optional.pdf").read_bytes()


def test_table_bundle_keeps_raw_values_and_bold_display_across_formats(tmp_path):
    raw = pd.DataFrame({"metric": [.123456789], "other": [.9]}, index=[7])
    display = pd.DataFrame({"AP": ["0.123 (0.100 to 0.150)"]}, index=[7])
    styles = {7: {"AP": {"bold": True, "bg": "#d1e7dd"}}}
    outputs = assets.export_dataframe_bundle(tmp_path / "compact", title="Comparison",
        dataframe=raw, display_dataframe=display, styles=styles)
    pd.testing.assert_frame_equal(pd.read_csv(outputs["csv"]), raw.reset_index(drop=True))
    assert "**0.123 (0.100 to 0.150)**" in Path(outputs["md"]).read_text()
    assert "font-weight: bold" in Path(outputs["html"]).read_text()
    assert b"/FontFile2" in Path(outputs["pdf"]).read_bytes()
    document = Document(outputs["docx"])
    assert document.tables[0].cell(1, 0).paragraphs[0].runs[0].bold
    borders = document.tables[0]._tbl.tblPr.xml
    assert 'w:insideV w:val="nil"' in borders
    plain = assets.export_dataframe_bundle(tmp_path / "complete", title="Complete", dataframe=raw)
    assert "html" in plain and "0.123456789" in Path(plain["html"]).read_text()


def test_html_export_preserves_values_escaping_ranks_and_numeric_alignment(tmp_path):
    frame = pd.DataFrame({"Model": ["EBM", "LR"], "Feature set": ["Clinical < waveform", "Preoperative"],
        "AP": ["0.123 (0.100 to 0.150)", None],
        "Δ AP": ["−0.002 (−0.010 to 0.005)", "Reference"],
        "Operations": [100, 100], "Comment": ["Prespecified", "Comparator"]}, index=[5, 6])
    output = assets.export_dataframe_bundle(tmp_path / "aligned", title="Primary < secondary",
        dataframe=frame, notes="95% CI < 1",
        styles={5: {"AP": {"bold": True, "bg": "#d1e7dd", "top_rule": True}}})
    html = Path(output["html"]).read_text()
    for column in (0, 1, 5):
        assert re.search(rf"td\.col{column}\s*\{{\s*text-align: left;", html)
    for column in (2, 3, 4):
        assert re.search(rf"td\.col{column}\s*\{{\s*text-align: right;\s*font-variant-numeric: tabular-nums;", html)
    assert "@media print" in html and "thead{display:table-header-group}" in html
    assert "Clinical &lt; waveform" in html and "Primary &lt; secondary" in html
    assert "95% CI &lt; 1" in html and "0.123 (0.100 to 0.150)" in html
    assert "font-weight: bold" in html and "border-top: 1px solid #777" in html
    assert ">nan<" not in html and ">None<" not in html
    prepared = pd.DataFrame({"Feature set": ["123", "Clinical only"],
        "AP": ["0.400 (0.300–0.500)", "Reference"],
        "Δ AP": ["−0.020 (−0.030–−0.010)", "0.010 (−0.005–0.020)"],
        "Finding": ["61.00 (53.00, 70.00)", "2,045 (57.3%)"],
        "Missing n (%)": ["0 (0.0%)", "6 (0.2%)"], "Notes": ["Observed", "Training only"]})
    assert assets.numeric_display_columns(prepared) == {"AP", "Δ AP", "Finding", "Missing n (%)"}


def test_all_metric_rank_styles_and_delta_directions_include_brier_and_f1():
    metrics = _metrics()
    styles = analysis.get_style_info(metrics)
    assert set(styles[1]) == set(analysis.METRIC_DISPLAY_MAP.values())
    assert all(cell["bold"] for cell in styles[1].values())
    delta = analysis.get_delta_style_info(metrics,
        {f"Δ {label}": f"delta_{key}" for key, label in analysis.METRIC_DISPLAY_MAP.items()})
    assert delta[0] == {}  # Reference intervals contain zero.
    assert set(delta[1]) == {f"Δ {label}" for label in analysis.METRIC_DISPLAY_MAP.values()}
    assert delta[1]["Δ Brier Score"]["bg"] == delta[1]["Δ F1 Score"]["bg"] == "#9fd4b2"
    boundary = pd.DataFrame({"delta_brier": [.02, -.02, -.02, np.nan],
        "delta_brier_lower": [.01, -.03, -.03, np.nan],
        "delta_brier_upper": [.03, -.01, 0, np.nan]})
    styled = analysis.get_delta_style_info(boundary, {"Δ Brier Score": "delta_brier"})
    assert styled[0]["Δ Brier Score"]["bg"] == "#f2b6bd"
    assert styled[1]["Δ Brier Score"]["bg"] == "#9fd4b2"
    assert styled[2] == styled[3] == {}
    tied = analysis.get_style_info(pd.DataFrame({"AUPRC_val": [.5, .5, np.nan]}))
    assert tied[0]["AUPRC"]["bold"] and tied[1]["AUPRC"]["bold"]
    assert not tied[2]["AUPRC"]["bold"]


def test_pdf_pagination_labels_panels_and_preserves_every_metric_and_row():
    metrics = list(analysis.METRIC_DISPLAY_MAP.values())
    frame = pd.DataFrame({"Model": ["EBM"] * 20,
        "Feature set": [f"Input {row}" for row in range(20)],
        **{metric: [f"{row / 100:.3f}" for row in range(20)] for metric in metrics}})
    pages = []
    observed = set()
    shaded = []

    class Capture:
        def savefig(self, figure, **kwargs):
            figure.canvas.draw()
            axis = figure.axes[0]
            table = axis.tables[0]
            renderer = figure.canvas.get_renderer()
            # A wrapped title plus pagination metadata stays above the table.
            assert figure.texts[0].get_window_extent(renderer).y0 > table.get_window_extent(renderer).y1
            headers = [" ".join(table[(0, column)].get_text().get_text().split())
                       for column in range(len({column for row, column in table.get_celld()}))]
            for cell in table.get_celld().values():
                if cell.get_facecolor() == to_rgba("#d1e7dd"):
                    shaded.append(cell)
                    assert cell.get_path().codes[-1] == PlotPath.CLOSEPOLY
                    assert cell.get_linewidth() == 0
            row_numbers = sorted({row for row, column in table.get_celld() if row})
            for row in row_numbers:
                identity = table[(row, 1)].get_text().get_text()
                for column, metric in enumerate(headers):
                    if metric in metrics:
                        observed.add((identity, metric, table[(row, column)].get_text().get_text()))
            pages.append((figure.texts[0].get_text(), headers, len(row_numbers)))

    assets._render_pdf_table_page(Capture(),
        "Complete postoperative outcome performance with saved patient-cluster intervals " * 2,
        frame, styles={0: {metrics[0]: {"bold": True, "bg": "#d1e7dd"}}})
    assert len(pages) == 4
    assert len(shaded) == 1
    for panel, metric_range in ((1, metrics[:5]), (2, metrics[5:])):
        matching = [page for page in pages if f"Panel {panel}/2" in page[0]]
        assert len(matching) == 2
        assert all(f"{metric_range[0]} to {metric_range[-1]}" in title
                   for title, headers, count in matching)
        assert any("Rows 1–18 of 20" in title and count == 18 for title, headers, count in matching)
        assert any("Rows 19–20 of 20" in title and count == 2 for title, headers, count in matching)
    expected = {(f"Input {row}", metric, f"{row / 100:.3f}")
                for row in range(20) for metric in metrics}
    assert observed == expected


def test_detailed_companion_retains_all_metrics_without_expanding_compact_table(tmp_path, monkeypatch):
    metrics = _metrics()
    before = metrics.copy(deep=True)
    assets.export_core_comparisons(metrics, tmp_path)
    stem = tmp_path / "manuscript_any_aki_main"
    compact = stem.with_suffix(".html").read_text()
    detailed = stem.with_name(stem.name + "_detailed")
    for suffix in ("html", "md", "docx", "pdf", "csv"):
        assert detailed.with_suffix(f".{suffix}").exists()
    main_csv = pd.read_csv(stem.with_suffix(".csv"))
    pd.testing.assert_frame_equal(main_csv, pd.read_csv(detailed.with_suffix(".csv")))
    for metric in analysis.METRIC_DISPLAY_MAP.values():
        assert metric in main_csv.columns
        assert metric in detailed.with_suffix(".html").read_text()
    assert "Sensitivity" not in compact
    delta_html = stem.with_name(stem.name + "_delta").with_suffix(".html").read_text()
    assert "Δ Precision" in delta_html and "Δ Accuracy" in delta_html
    pd.testing.assert_frame_equal(metrics, before)

    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr(analysis, "TABLES_DIR", tmp_path / "legacy")
    monkeypatch.setattr(analysis, "PAPER_DIR", tmp_path)
    analysis.generate_html_tables(metrics)
    base = tmp_path / "legacy" / "results_any_aki_non_windowed_EBM"
    assert "**0.500 (0.490-0.510)**" in base.with_suffix(".md").read_text()
    html = base.with_suffix(".html").read_text()
    assert "Precision" in html and "Accuracy" in html and "font-weight: bold" in html


@pytest.mark.parametrize('with_split', [False, True])
def test_publication_cohort_flow_keeps_counts_and_labels_readable_at_native_size(with_split):
    from reporting.cohort_flow import CohortStage, OutcomeSplit, _draw_cohort_flow_fallback
    stages = [CohortStage(title=title, count=count, n_patients=patients,
                         removed=removed, removal_reason=reason)
              for title,count,patients,removed,reason in [
        ('VitalDB source operations',6388,6090,None,None),
        ('General anesthesia',6043,5783,345,'Non-general anesthesia'),
        ('ASA eligibility',6031,5781,12,'ASA V/VI'),
        ('All four waveforms available',5779,5557,252,'Required waveform unavailable'),
        ('Shared AKI/ICU eligibility',3581,3442,2198,'AKI ascertainment requirements not met'),
        ('Final analysis cohort',3571,3432,10,'No retained waveform-feature record')]]
    split = OutcomeSplit('AKI',189,3382) if with_split else None
    fig = _draw_cohort_flow_fallback(stages,split,None,publication=True)
    try:
        fig.canvas.draw()
        assert fig.get_size_inches()[0]*25.4 == pytest.approx(174)
        ax=fig.axes[0];renderer=fig.canvas.get_renderer()
        labels=[t for t in ax.texts if t.get_text() and t.get_text()!='Excluded operations']
        assert len(labels)==len(ax.patches)
        for i,(label,patch) in enumerate(zip(labels,ax.patches)):
            text=label.get_window_extent(renderer);box=patch.get_window_extent(renderer)
            assert box.x0<=text.x0<text.x1<=box.x1
            assert box.y0<=text.y0<text.y1<=box.y1
            assert all(not box.overlaps(p.get_window_extent(renderer)) for p in ax.patches[i+1:])
            assert 8<=label.get_fontsize()<=12
        text='\n'.join(t.get_text() for t in labels)
        for stage in stages:
            assert f'n = {stage.count:,} operations' in text
            assert f'{stage.n_patients:,} unique patients' in text
            if stage.removed:
                assert f'n = {stage.removed:,}' in text
                assert stage.removal_reason.replace(' ','') in text.replace('\n','').replace(' ','')
        assert sum(s.removed or 0 for s in stages)==stages[0].count-stages[-1].count
    finally:
        plt.close(fig)
