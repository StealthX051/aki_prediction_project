import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from docx import Document

from model_creation.preprocessing import FoldPreprocessor
import results_recreation.metrics_summary as metrics_summary
import results_recreation.results_analysis as results_analysis
import reporting.make_shap_figures as shap_figures


def test_reporting_help_imports_without_creating_output_directories(tmp_path):
    environment = os.environ.copy()
    destinations = {
        "RESULTS_DIR": tmp_path / "experiments",
        "PAPER_DIR": tmp_path / "paper",
        "FIGURES_DIR": tmp_path / "paper/figures",
        "TABLES_DIR": tmp_path / "paper/tables",
        "REPORTS_DIR": tmp_path / "paper/reports",
    }
    environment.update({name: str(path) for name, path in destinations.items()})
    environment["AKI_STORAGE_POLICY"] = "enforce"
    environment["LOG_FILE"] = str(tmp_path / "imports.log")
    result = subprocess.run(
        [sys.executable, "-m", "reporting.make_report", "--help"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, check=True,
    )
    assert "--strict" in result.stdout and "--run-manifest" in result.stdout
    assert all(not path.exists() for path in destinations.values())


def test_report_writers_enforce_storage_policy_before_creating_outputs(tmp_path, monkeypatch):
    from artifact_paths import StoragePolicyError
    monkeypatch.setenv("AKI_STORAGE_POLICY", "enforce")
    for name in ("TABLES_DIR", "REPORTS_DIR", "FIGURES_DIR"):
        monkeypatch.setattr(results_analysis, name, tmp_path / name.lower())
    metrics = results_analysis.prepare_metrics_for_display(_summary_df())
    for write in (results_analysis.generate_html_tables, results_analysis.generate_markdown_report,
                  results_analysis.generate_docx_report, results_analysis.generate_pdf_report):
        with pytest.raises(StoragePolicyError):
            write(metrics)
    predictions = pd.DataFrame({"outcome": "any_aki", "branch": "windowed", "model_name": "XGBoost",
        "feature_set": "preop_only", "y_true": [0, 1], "y_prob_calibrated": [.1, .8]})
    with pytest.raises(StoragePolicyError):
        results_analysis.plot_curves(predictions, results_analysis.PlotConfig(n_jobs=1))
    assert not list(tmp_path.iterdir())


@pytest.fixture
def fresh_shap_caches():
    caches = (shap_figures._branch_dataset, shap_figures._display_dictionary)
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:
        cached.cache_clear()


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

    expected = _summary_df()
    metrics_summary.write_outputs(expected, bootstrap_df=None, save_bootstrap=None)

    for directory in (tables_dir, paper_tables_dir):
        pd.testing.assert_frame_equal(pd.read_csv(directory / "metrics_summary.csv"), expected)
    assert "0.810" in (paper_tables_dir / "metrics_summary.md").read_text()
    document = Document(paper_tables_dir / "metrics_summary.docx")
    cells = " ".join(cell.text for table in document.tables for row in table.rows for cell in row.cells)
    assert "0.810" in cells and "0.620" in cells
    assert (paper_tables_dir / "metrics_summary.pdf").read_bytes().startswith(b"%PDF-")


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

    from reporting.make_report import _report_context_notes
    summary = _summary_df().assign(outcome="mortality", analysis_status="definitive")
    metrics_df = results_analysis.prepare_metrics_for_display(summary)
    notes = _report_context_notes({"analysis_role": "exploratory", "branch_rosters": {
        "windowed": {"operations": 5765, "patients": 5543,
                     "outcomes": {"mortality": {"events": 52, "event_patients": 47}}}}})
    metrics_df.attrs["report_notes"] = notes
    pdf_text = {}
    original_savefig = results_analysis.PdfPages.savefig
    def inspect_pdf(self, figure, *args, **kwargs):
        from matplotlib.text import Text
        pdf_text.setdefault(Path(self._filename).name, []).append(
            "\n".join(text.get_text() for text in figure.findobj(Text)))
        return original_savefig(self, figure, *args, **kwargs)
    monkeypatch.setattr(results_analysis.PdfPages, "savefig", inspect_pdf)
    results_analysis.generate_html_tables(metrics_df)
    results_analysis.generate_markdown_report(metrics_df)
    results_analysis.generate_docx_report(metrics_df)
    results_analysis.generate_pdf_report(metrics_df)

    base = tables_dir / "results_mortality_windowed_XGBoost"
    for suffix in (".html", ".md"):
        assert "0.810" in base.with_suffix(suffix).read_text()
    document = Document(base.with_suffix(".docx"))
    assert any("0.810" in cell.text for row in document.tables[0].rows for cell in row.cells)
    assert base.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")
    main = pd.read_csv(tables_dir / "results_mortality_windowed_XGBoost_main.csv")
    delta = pd.read_csv(tables_dir / "results_mortality_windowed_XGBoost_delta.csv")
    assert main.loc[0, "AUROC"] == "0.810 (0.750-0.870)"
    assert delta.loc[0, "Δ Brier Score"] == "-0.020 (-0.030--0.010)"

    assert "0.810" in (reports_dir / "report.md").read_text()
    report = Document(reports_dir / "report.docx")
    assert any("0.810" in cell.text for table in report.tables for row in table.rows for cell in row.cells)
    assert (reports_dir / "report.pdf").read_bytes().startswith(b"%PDF-")
    context = ("Scientific analysis role: exploratory", "5765 operations; 5543 patients",
               "Mortality-positive operations: 52; positive patients: 47")
    for text in context:
        assert text in base.with_suffix(".html").read_text()
        assert text in (reports_dir / "report.md").read_text()
        assert any(text in paragraph.text for paragraph in report.paragraphs)
        assert any(text in page for page in pdf_text["report.pdf"])
        assert any(text in paragraph.text for paragraph in document.paragraphs)

    manifest_path = paper_dir / "manifest.json"
    readme_path = paper_dir / "README.md"
    assert manifest_path.exists()
    assert readme_path.exists()
    assert "results_mortality_windowed_XGBoost" in manifest_path.read_text(encoding="utf-8")
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
        assert (figures_dir / "primary_figures" / f"{stem}.png").exists()
        assert (figures_dir / "primary_figures" / f"{stem}.svg").exists()
        image = plt.imread(figures_dir / f"{stem}.png")
        assert image.ndim == 3 and min(image.shape[:2]) > 100
        assert np.isfinite(image).all() and np.ptp(image[:, :, :3]) > 0

    # The exported curves must retain the known perfect-ranking scores, not
    # merely create files with the expected names.
    assert "AUC = 1.000" in (figures_dir / "roc_any_aki_windowed_XGBoost.svg").read_text()
    assert "AP = 1.000" in (figures_dir / "pr_any_aki_windowed_XGBoost.svg").read_text()
    calibration = (figures_dir / "calibration_any_aki_windowed_XGBoost.svg").read_text()
    assert "Mean Predicted Probability" in calibration and "Fraction of Positives" in calibration


def test_supplemental_fold_variability_is_numerically_correct_and_not_labeled_as_ci(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    paper_dir = tmp_path / "paper"
    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(results_analysis, "PAPER_DIR", paper_dir)
    monkeypatch.setattr(results_analysis, "FIGURES_DIR", figures_dir)

    prediction_df = pd.DataFrame(
        {
            "outcome": ["any_aki"] * 8,
            "branch": ["windowed"] * 8,
            "model_name": ["XGBoost"] * 8,
            "feature_set": ["preop_only"] * 8,
            "y_true": [0, 1, 0, 1, 0, 1, 0, 1],
            "y_prob_calibrated": [0.1, 0.8, 0.2, 0.9, 0.7, 0.6, 0.2, 0.8],
            "repeat_id": [0] * 8,
            "outer_fold_id": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    figures = {}
    def inspect_figure(fig, path, **kwargs):
        ax = fig.axes[0]
        figures[path.stem] = (ax.get_title(), [line.get_label() for line in ax.lines], len(ax.collections))
        plt.close(fig)
        return []
    monkeypatch.setattr(results_analysis, "save_report_figure_bundle", inspect_figure)
    results_analysis.plot_curves(prediction_df, config=results_analysis.PlotConfig(n_jobs=1, curve_mode="fold_variability"))

    # Fold AUROCs are 1 and .75: mean .875, population SD .125.
    title, labels, bands = figures["roc_any_aki_windowed_XGBoost"]
    assert "AUC = 0.875 +/- 0.125" in labels[0]
    assert "fold mean ± SD; not a confidence interval" in title
    assert bands == 1
    assert figures["pr_any_aki_windowed_XGBoost"][2] == 1


def test_saved_xgboost_shap_figures_preserve_heldout_features_without_refitting(tmp_path: Path, monkeypatch, fresh_shap_caches):
    import xgboost as xgb

    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")

    results_dir = tmp_path / "results" / "catch22" / "experiments"
    paper_dir = tmp_path / "results" / "catch22" / "paper"
    figures_dir = paper_dir / "figures"
    scatter_dir = figures_dir / "shap_scatter"
    featured_dir = figures_dir / "shap_scatter_featured"
    model_dir = results_dir / "models" / "xgboost" / "any_aki" / "non_windowed" / "preop_only"
    predictions_dir = model_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dataset_df = pd.DataFrame(
        {
            "caseid": np.arange(1, 9),
            "subjectid": np.arange(101, 109),
            "age": [45, 51, 60, 67, 58, 72, 49, 63],
            "preop_cr": [0.8, 1.1, 1.5, 1.8, 1.0, 1.7, 0.9, 1.4],
            "preop_hb": [13.9, 12.8, 11.2, 10.5, 13.1, 10.8, 14.2, 11.7],
            "sex": ["F", "M", "M", "F", "F", "M", "F", "M"],
            "any_aki": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    dataset_path = tmp_path / "aki_features_master_wide.csv"
    dataset_df.to_csv(dataset_path, index=False)

    feature_df = dataset_df[["age", "preop_cr", "preop_hb", "sex"]].copy()
    y = dataset_df["any_aki"].astype(int)
    preprocessor = FoldPreprocessor(impute_missing=False)
    train_rows = [0, 1, 6, 7]
    preprocessor.fit(feature_df.iloc[train_rows])
    transformed = preprocessor.transform(feature_df)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=12,
        max_depth=2,
        min_child_weight=0,  # Allow a nonconstant tree on this four-row fixture.
        learning_rate=0.3,
        random_state=0,
        n_jobs=1,
    )
    model.fit(transformed.iloc[train_rows], y.iloc[train_rows])
    model.save_model(model_dir / "model.json")
    with (model_dir / "preprocessor.pkl").open("wb") as fp:
        pickle.dump(preprocessor, fp)

    test_rows = dataset_df.iloc[[2, 3, 4, 5]].copy().reset_index(drop=True)
    transformed_test = preprocessor.transform(test_rows[["age", "preop_cr", "preop_hb", "sex"]])
    y_prob = model.predict_proba(transformed_test)[:, 1]
    predictions_df = pd.DataFrame(
        {
            "caseid": test_rows["caseid"],
            "y_true": test_rows["any_aki"],
            "y_prob_raw": y_prob,
            "y_prob_calibrated": y_prob,
            "threshold": [0.5] * len(test_rows),
            "y_pred_label": (y_prob >= 0.5).astype(int),
            "fold": [0] * len(test_rows),
            "is_oof": [False] * len(test_rows),
            "outcome": ["any_aki"] * len(test_rows),
            "branch": ["non_windowed"] * len(test_rows),
            "feature_set": ["preop_only"] * len(test_rows),
            "model_name": ["XGBoost"] * len(test_rows),
            "pipeline": ["catch22"] * len(test_rows),
        }
    )
    predictions_df.to_csv(predictions_dir / "test.csv", index=False)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = model_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "calibration.json").write_text(json.dumps({"intercept": 0.0, "slope": 1.0, "eps": 1e-15}))
    (artifacts_dir / "threshold.json").write_text(json.dumps({"threshold": 0.5}))

    monkeypatch.setattr(shap_figures, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(shap_figures, "PAPER_DIR", paper_dir)
    monkeypatch.setattr(shap_figures, "FIGURES_DIR", figures_dir)
    monkeypatch.setattr(shap_figures, "SHAP_SCATTER_DIR", scatter_dir)
    monkeypatch.setattr(shap_figures, "FEATURED_SCATTER_DIR", featured_dir)
    monkeypatch.setattr(shap_figures.model_utils, "load_data", lambda branch: pd.read_csv(dataset_path))
    monkeypatch.setattr(xgb.XGBClassifier, "fit", lambda *a, **kw: pytest.fail("SHAP export must not refit"))
    monkeypatch.setattr(FoldPreprocessor, "fit", lambda *a, **kw: pytest.fail("SHAP export must not fit preprocessing"))

    bundles = []
    original_bundle = shap_figures._bundle_for_job
    def capture_bundle(*args, **kwargs):
        bundle = original_bundle(*args, **kwargs)
        bundles.append(bundle)
        return bundle
    monkeypatch.setattr(shap_figures, "_bundle_for_job", capture_bundle)

    outputs = shap_figures.generate_shap_outputs(n_jobs=1)

    assert outputs
    assert bundles[0].predictions.caseid.tolist() == test_rows.caseid.tolist()
    assert set(bundles[0].raw_test_frame.caseid).isdisjoint(dataset_df.iloc[train_rows].caseid)
    importance = pd.read_csv(figures_dir / "shap_importance_any_aki_non_windowed_preop_only.csv")
    assert set(importance.feature) == set(transformed_test.columns)
    assert np.isfinite(importance.mean_abs_shap).all()
    assert importance.mean_abs_shap.max() > 0
    expected = np.abs(bundles[0].shap_values).mean(axis=0)
    np.testing.assert_allclose(importance.set_index("feature").loc[transformed_test.columns, "mean_abs_shap"], expected)
    assert (figures_dir / "shap_beeswarm_any_aki_non_windowed_preop_only.png").exists()
    assert (figures_dir / "shap_beeswarm_any_aki_non_windowed_preop_only.svg").exists()
    assert (figures_dir / "primary_figures" / "shap_beeswarm_any_aki_non_windowed_preop_only.png").exists()
    assert list(scatter_dir.glob("shap_scatter_any_aki_non_windowed_preop_only_*.png"))
    assert list(scatter_dir.glob("shap_scatter_any_aki_non_windowed_preop_only_*.svg"))
    assert list(featured_dir.glob("shap_scatter_any_aki_non_windowed_preop_only_*.png"))
