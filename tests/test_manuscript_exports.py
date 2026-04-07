from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import results_recreation.metrics_summary as metrics_summary
import results_recreation.results_analysis as results_analysis


def _summary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "outcome": "any_aki",
                "branch": "windowed",
                "feature_set": "preop_only",
                "model_name": "XGBoost",
                "auroc": 0.81,
                "auroc_lower": 0.75,
                "auroc_upper": 0.87,
                "auprc": 0.62,
                "auprc_lower": 0.54,
                "auprc_upper": 0.70,
                "brier": 0.18,
                "brier_lower": 0.15,
                "brier_upper": 0.21,
                "sensitivity": 0.72,
                "sensitivity_lower": 0.61,
                "sensitivity_upper": 0.83,
                "specificity": 0.79,
                "specificity_lower": 0.70,
                "specificity_upper": 0.88,
                "f1": 0.69,
                "f1_lower": 0.58,
                "f1_upper": 0.80,
                "delta_auroc": 0.05,
                "delta_auroc_lower": 0.01,
                "delta_auroc_upper": 0.09,
                "delta_auprc": 0.04,
                "delta_auprc_lower": 0.01,
                "delta_auprc_upper": 0.07,
                "delta_brier": -0.02,
                "delta_brier_lower": -0.03,
                "delta_brier_upper": -0.01,
                "delta_sensitivity": 0.03,
                "delta_sensitivity_lower": 0.0,
                "delta_sensitivity_upper": 0.06,
                "delta_specificity": 0.02,
                "delta_specificity_lower": 0.0,
                "delta_specificity_upper": 0.05,
                "delta_f1": 0.04,
                "delta_f1_lower": 0.01,
                "delta_f1_upper": 0.07,
            }
        ]
    )


def test_metrics_summary_write_outputs_exports_manuscript_bundle(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    tables_dir = tmp_path / "results" / "catch22" / "experiments" / "tables"
    paper_dir = tmp_path / "results" / "catch22" / "paper"
    paper_tables_dir = paper_dir / "tables"

    monkeypatch.setattr(metrics_summary, "TABLES_DIR", tables_dir)
    monkeypatch.setattr(metrics_summary, "PAPER_DIR", paper_dir)
    monkeypatch.setattr(metrics_summary, "PAPER_TABLES_DIR", paper_tables_dir)

    metrics_summary.write_outputs(_summary_df(), bootstrap_df=None, save_bootstrap=None)

    assert (tables_dir / "metrics_summary.csv").exists()
    assert (paper_tables_dir / "metrics_summary.csv").exists()
    assert (paper_tables_dir / "metrics_summary.md").exists()
    assert (paper_tables_dir / "metrics_summary.docx").exists()
    assert (paper_tables_dir / "metrics_summary.pdf").exists()


def test_results_analysis_exports_table_bundles_and_reports(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    paper_dir = tmp_path / "paper"
    tables_dir = paper_dir / "tables"
    reports_dir = paper_dir / "reports"
    figures_dir = paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(results_analysis, "PAPER_DIR", paper_dir)
    monkeypatch.setattr(results_analysis, "TABLES_DIR", tables_dir)
    monkeypatch.setattr(results_analysis, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(results_analysis, "FIGURES_DIR", figures_dir)

    metrics_df = results_analysis.prepare_metrics_for_display(_summary_df())
    results_analysis.generate_html_tables(metrics_df)
    results_analysis.generate_markdown_report(metrics_df)
    results_analysis.generate_docx_report(metrics_df)
    results_analysis.generate_pdf_report(metrics_df)

    base = tables_dir / "results_any_aki_windowed_XGBoost"
    assert base.with_suffix(".html").exists()
    assert base.with_suffix(".md").exists()
    assert base.with_suffix(".docx").exists()
    assert base.with_suffix(".pdf").exists()
    assert (tables_dir / "results_any_aki_windowed_XGBoost_main.csv").exists()
    assert (tables_dir / "results_any_aki_windowed_XGBoost_delta.csv").exists()

    assert (reports_dir / "report.md").exists()
    assert (reports_dir / "report.docx").exists()
    assert (reports_dir / "report.pdf").exists()

    manifest_path = paper_dir / "manifest.json"
    readme_path = paper_dir / "README.md"
    assert manifest_path.exists()
    assert readme_path.exists()
    assert "results_any_aki_windowed_XGBoost" in manifest_path.read_text(encoding="utf-8")
    assert "Core reports" in readme_path.read_text(encoding="utf-8")


def test_plot_curves_exports_svg_and_png(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    paper_dir = tmp_path / "paper"
    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(results_analysis, "PAPER_DIR", paper_dir)
    monkeypatch.setattr(results_analysis, "FIGURES_DIR", figures_dir)

    prediction_df = pd.DataFrame(
        {
            "outcome": ["any_aki"] * 4,
            "branch": ["windowed"] * 4,
            "model_name": ["XGBoost"] * 4,
            "feature_set": ["preop_only"] * 4,
            "y_true": [0, 1, 0, 1],
            "y_prob_calibrated": [0.1, 0.8, 0.2, 0.9],
        }
    )

    results_analysis.plot_curves(prediction_df, config=results_analysis.PlotConfig(n_jobs=1))

    for stem in (
        "roc_any_aki_windowed_XGBoost",
        "pr_any_aki_windowed_XGBoost",
        "calibration_any_aki_windowed_XGBoost",
    ):
        assert (figures_dir / f"{stem}.png").exists()
        assert (figures_dir / f"{stem}.svg").exists()
