import json
from pathlib import Path
import pickle
import sys

import matplotlib.axes
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_creation.preprocessing import FoldPreprocessor
import results_recreation.metrics_summary as metrics_summary
import results_recreation.results_analysis as results_analysis
import reporting.make_shap_figures as shap_figures


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
        assert (figures_dir / "primary_figures" / f"{stem}.png").exists()
        assert (figures_dir / "primary_figures" / f"{stem}.svg").exists()


def test_plot_curves_uses_foldwise_uncertainty_bands_when_fold_metadata_present(tmp_path: Path, monkeypatch):
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
            "y_prob_calibrated": [0.15, 0.82, 0.20, 0.77, 0.10, 0.88, 0.24, 0.81],
            "repeat_id": [0] * 8,
            "outer_fold_id": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    fill_between_calls = {"count": 0}
    original_fill_between = matplotlib.axes.Axes.fill_between

    def _count_fill_between(self, *args, **kwargs):
        fill_between_calls["count"] += 1
        return original_fill_between(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", _count_fill_between)
    results_analysis.plot_curves(prediction_df, config=results_analysis.PlotConfig(n_jobs=1))

    assert fill_between_calls["count"] >= 2


def test_make_shap_figures_generates_beeswarm_and_scatter_outputs(tmp_path: Path, monkeypatch):
    pytest.importorskip("xgboost")
    pytest.importorskip("shap")
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
    transformed = preprocessor.fit_transform(feature_df)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=12,
        max_depth=2,
        learning_rate=0.3,
        random_state=0,
        n_jobs=1,
    )
    model.fit(transformed, y)
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
    shap_figures._branch_dataset.cache_clear()
    shap_figures._display_dictionary.cache_clear()
    monkeypatch.setattr(shap_figures.model_utils, "load_data", lambda branch: pd.read_csv(dataset_path))

    outputs = shap_figures.generate_shap_outputs(n_jobs=1)

    assert outputs
    assert (figures_dir / "shap_beeswarm_any_aki_non_windowed_preop_only.png").exists()
    assert (figures_dir / "shap_beeswarm_any_aki_non_windowed_preop_only.svg").exists()
    assert (figures_dir / "primary_figures" / "shap_beeswarm_any_aki_non_windowed_preop_only.png").exists()
    assert list(scatter_dir.glob("shap_scatter_any_aki_non_windowed_preop_only_*.png"))
    assert list(scatter_dir.glob("shap_scatter_any_aki_non_windowed_preop_only_*.svg"))
    assert list(featured_dir.glob("shap_scatter_any_aki_non_windowed_preop_only_*.png"))
