"""Reporting consumes verified comparison statistics without fitting or resampling."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reporting import make_report
from reporting import decision_curve
from results_recreation import metrics_summary as metrics


def _report_fixture(tmp_path, monkeypatch):
    frame = pd.DataFrame({"caseid": [1, 2, 3, 4], "subjectid": [10, 20, 30, 40],
        "y_true": [0, 1, 0, 1], "y_prob_calibrated": [.1, .8, .2, .7],
        "y_pred_label": [0, 1, 0, 1], "threshold": .5, "outer_fold_id": [0, 0, 1, 1]})
    ps = metrics.PredictionSet(tmp_path / "predictions/test.csv", frame, .5,
        "any_aki", "non_windowed", "preop_only", "XGBoost", "nested", "main_shared", "nested_cv", "diagnostic")
    summary, _ = metrics.summarize([ps], n_bootstrap=0, strict=True)
    monkeypatch.setattr(make_report, "crawl_predictions", lambda *args, **kwargs: [ps])
    monkeypatch.setattr(make_report, "_load_metrics", lambda *args: summary)
    for name in ("TABLES_DIR", "FIGURES_DIR", "REPORTS_DIR", "RESULTS_DIR"):
        monkeypatch.setattr(make_report, name, tmp_path / name.lower())
    return ps, summary


def _saved_comparisons():
    pair = {"cohort_scope": "main_shared", "validation_scheme": "nested_cv", "outcome": "any_aki",
        "branch": "non_windowed", "comparison_id": "pair", "comparison_type": "input", "designation": "primary",
        "reference_model": "xgboost", "candidate_model": "xgboost", "reference_feature_set": "preop_only",
        "candidate_feature_set": "preop_and_all_waveforms", "metric": "auprc", "reference_estimate": .2,
        "candidate_estimate": .23, "delta": .03, "lower": -.01, "upper": .07, "bootstrap_valid": 50,
        "n_operations": 4, "n_patients": 4, "analysis_status": "diagnostic"}
    return {"paired": pd.DataFrame([pair]), "dca": pd.DataFrame({"saved": [1]}),
        "dca_comparisons": pd.DataFrame({"saved": [2]}),
        "delong": pd.DataFrame({"status": ["unsupported_design"]}),
        "inventory": {"comparison_status": "partial", "comparisons": [
            {"availability": "available"}, {"availability": "not_requested"}]}}


def test_mortality_report_role_is_exploratory_even_with_full_validation():
    manifest = {"cohort_scope": "mortality_waveform_sensitivity", "diagnostic": False}
    assert "Scientific analysis role: exploratory" in make_report._report_context_notes(manifest)[0]
    assert make_report._analysis_role(manifest) == "exploratory"
    assert make_report._analysis_role({"cohort_scope": "icu_waveform_sensitivity"}) == "sensitivity"
    with pytest.raises(ValueError, match="exploratory scientific role"):
        make_report._report_context_notes({**manifest, "analysis_role": "primary"})


def _disable_legacy_exports(monkeypatch):
    for name in ("generate_html_tables", "generate_markdown_report", "generate_docx_report", "generate_pdf_report"):
        monkeypatch.setattr(make_report, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(make_report, "prepare_metrics_for_display", lambda frame: frame)
    import reporting.manuscript_assets as assets
    monkeypatch.setattr(assets, "export_core_comparisons", lambda *args, **kwargs: None)
    monkeypatch.setattr(assets, "refresh_paper_bundle", lambda *args, **kwargs: None)
    monkeypatch.setattr(make_report, "plot_curves", lambda *args, **kwargs: None)


def test_strict_report_reads_saved_comparisons_and_labels_partial_coverage(tmp_path, monkeypatch):
    _report_fixture(tmp_path, monkeypatch)
    artifacts = _saved_comparisons()
    monkeypatch.setattr(make_report, "_load_comparisons", lambda *args, **kwargs: artifacts)
    _disable_legacy_exports(monkeypatch)
    make_report.RESULTS_DIR.mkdir()
    (make_report.RESULTS_DIR / "run_manifest.json").write_text(json.dumps({
        "analysis_role": "exploratory", "branch_rosters": {
            "non_windowed": {"operations": 4, "patients": 4}}}))
    exported_context = []
    for name in ("generate_html_tables", "generate_markdown_report", "generate_docx_report", "generate_pdf_report"):
        monkeypatch.setattr(make_report, name, lambda frame: exported_context.append(frame.attrs["report_notes"]))
    tables, plots = [], []
    comparison_context = []
    monkeypatch.setattr(make_report, "_export_comparison_tables", lambda data, path, **kwargs:
                        (tables.append(data), comparison_context.append(kwargs["report_notes"])))
    monkeypatch.setattr(decision_curve, "render_comparison_curves", lambda data, pairs, path, **kwargs:
                        plots.append((data, pairs, kwargs)), raising=False)
    monkeypatch.setattr(decision_curve, "net_benefit", lambda *args, **kwargs: pytest.fail("Report must not recalculate DCA"))
    monkeypatch.setattr(decision_curve, "export_decision_curves", lambda *args, **kwargs: pytest.fail("Legacy DCA path used"))
    monkeypatch.setattr(metrics, "summarize", lambda *args, **kwargs: pytest.fail("Report must not resample metrics"))
    monkeypatch.setattr(metrics, "_generate_bootstrap_indices", lambda *args, **kwargs: pytest.fail("Report must not generate draws"))
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", lambda *args, **kwargs: pytest.fail("Report must not fit models/calibration"))
    monkeypatch.setattr(sys, "argv", ["make_report", "--strict"])
    make_report.main()
    assert tables == [artifacts]
    assert plots[0][0] is artifacts["dca"] and plots[0][1] is artifacts["dca_comparisons"]
    assert plots[0][2]["comparison_status"] == "partial"
    status = json.loads((make_report.REPORTS_DIR / "report_status.json").read_text())
    assert status["comparison_status"] == "partial"
    assert status["comparison_coverage"] == {"available": 1, "not_requested": 1}
    assert status["delong_status_counts"] == {"unsupported_design": 1}
    assert status["analysis_role"] == "exploratory"
    assert len(exported_context) == 4
    assert all("Scientific analysis role: exploratory" in notes[0] and "4 operations; 4 patients" in notes[1]
               for notes in exported_context)
    assert comparison_context == [exported_context[0]]


def test_strict_comparison_validation_precedes_every_export(tmp_path, monkeypatch):
    _report_fixture(tmp_path, monkeypatch)
    reason = "Invalid comparison provenance"
    def reject(*args, **kwargs):
        raise ValueError(reason)
    monkeypatch.setattr(make_report, "_load_comparisons", reject)
    monkeypatch.setattr(make_report, "prepare_metrics_for_display", lambda *args: pytest.fail("Export happened before validation"))
    monkeypatch.setattr(sys, "argv", ["make_report", "--strict"])
    with pytest.raises(ValueError, match=reason):
        make_report.main()
    assert not make_report.TABLES_DIR.exists() and not make_report.REPORTS_DIR.exists()


def test_legacy_report_explicitly_uses_point_curve_fallback(tmp_path, monkeypatch):
    _report_fixture(tmp_path, monkeypatch)
    _disable_legacy_exports(monkeypatch)
    monkeypatch.setattr(make_report, "_load_comparisons", lambda *args, **kwargs: pytest.fail("Legacy mode requested strict artifacts"))
    calls = []
    monkeypatch.setattr(decision_curve, "export_decision_curves", lambda *args, **kwargs: calls.append(True))
    monkeypatch.setattr(sys, "argv", ["make_report", "--legacy"])
    make_report.main()
    assert calls == [True]


def test_comparison_table_export_preserves_negative_and_uncertain_differences(tmp_path, monkeypatch):
    import reporting.manuscript_assets as assets
    artifacts = _saved_comparisons()
    artifacts["paired"].loc[0, ["delta", "lower", "upper"]] = [-.03, -.07, .01]
    captured = []
    monkeypatch.setattr(assets, "export_dataframe_bundle", lambda *args, **kwargs: captured.append(kwargs))
    make_report._export_comparison_tables(artifacts, tmp_path, report_notes=["Scientific analysis role: exploratory."])
    display = captured[0]["display_dataframe"]
    assert display["Difference (95% interval)"].iloc[0] == "-0.030 (-0.070, 0.010)"
    assert "[partial]" in captured[0]["title"]
    assert "[diagnostic]" in captured[0]["title"]
    assert "candidate minus reference" in captured[0]["notes"]
    assert "Scientific analysis role: exploratory." in captured[0]["notes"]


def test_comparison_tables_keep_prespecified_family_and_metric_order(tmp_path, monkeypatch):
    import reporting.manuscript_assets as assets
    artifacts = _saved_comparisons()
    base = artifacts["paired"].iloc[0].to_dict()
    records = []
    pairs = [("logistic_regression", "xgboost"), ("logistic_regression", "ebm"), ("xgboost", "ebm")]
    for reference, candidate in pairs:
        for metric in ("auprc", "auroc", "brier"):
            records.append({**base, "reference_model": reference, "candidate_model": candidate,
                            "comparison_type": "family", "designation": "secondary", "metric": metric})
    artifacts["paired"] = pd.DataFrame(records).sample(frac=1, random_state=42)
    captured = []
    monkeypatch.setattr(assets, "export_dataframe_bundle", lambda *args, **kwargs: captured.append(kwargs))
    make_report._export_comparison_tables(artifacts, tmp_path)
    ordered = captured[0]["dataframe"]
    assert list(zip(ordered.reference_model, ordered.candidate_model)) == [pair for pair in pairs for _ in range(3)]
    assert ordered.metric.tolist() == ["auprc", "auroc", "brier"] * 3


def test_actual_metrics_to_verified_saved_comparison_report_pipeline(tmp_path, monkeypatch):
    """Exercise the real statistics writer/loader boundary without model fitting."""
    from results_recreation import comparisons
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from interpret.glassbox import ExplainableBoostingClassifier

    results, paper = tmp_path / "results", tmp_path / "paper"
    tables = results / "tables"
    truth = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    labels = {"logistic_regression": "Logistic regression", "xgboost": "XGBoost", "ebm": "EBM"}
    predictions, configurations = [], []
    for i, family in enumerate(comparisons.FAMILIES):
        for j, feature in enumerate(comparisons.CORE):
            probabilities = np.clip(np.array([.2, .45, .4, .65, .6, .5, .3, .7])
                                    + (truth - .5) * (i + j) * .07, .01, .99)
            frame = pd.DataFrame({"caseid": np.arange(8), "subjectid": [10, 10, 20, 30, 40, 40, 50, 60],
                "y_true": truth, "y_prob_raw": probabilities, "y_prob_calibrated": probabilities,
                "y_pred_label": (probabilities >= .5).astype(int), "threshold": .5,
                "outer_fold_id": [0, 0, 1, 1, 0, 0, 1, 1], "repeat_id": 0})
            path = results / "models" / family / "any_aki/non_windowed" / feature / "predictions/test.csv"
            path.parent.mkdir(parents=True)
            frame.to_csv(path, index=False)
            predictions.append(metrics.PredictionSet(path, pd.read_csv(path), .5, "any_aki", "non_windowed",
                feature, labels[family], "nested", "main_shared", "nested_cv", "diagnostic"))
            configurations.append({"model_type": family, "outcome": "any_aki",
                                   "branch": "non_windowed", "feature_set": feature})
    manifest = results / "run_manifest.json"
    manifest.write_text(json.dumps({"cohort_scope": "main_shared", "validation_scheme": "nested_cv",
        "diagnostic": True, "configurations": configurations,
        "reporting": {"bootstrap_replicates": 8, "bootstrap_seed": 17, "bootstrap_stratified": False}}))
    for cls in (LogisticRegression, XGBClassifier, ExplainableBoostingClassifier):
        monkeypatch.setattr(cls, "fit", lambda *args, **kwargs: pytest.fail("Statistics/reporting must never fit models"))
    monkeypatch.setattr(metrics, "crawl_predictions", lambda *args, **kwargs: predictions)
    monkeypatch.setattr(make_report, "crawl_predictions", lambda *args, **kwargs: predictions)
    for name, value in (("RESULTS_DIR", results), ("TABLES_DIR", tables), ("PAPER_DIR", paper),
                        ("PAPER_TABLES_DIR", paper / "tables")):
        monkeypatch.setattr(metrics, name, value)
    monkeypatch.setattr(metrics, "export_dataframe_bundle", lambda *args, **kwargs: None)
    monkeypatch.setattr(metrics, "refresh_paper_bundle", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["metrics_summary", "--results-dir", str(results), "--run-manifest", str(manifest),
        "--strict", "--bootstrap", "8", "--bootstrap-seed", "17", "--no-bootstrap-stratified",
        "--delta-mode", "reference", "--parallel-backend", "threads", "--n-jobs", "1"])
    metrics.main()
    inventory = json.loads((tables / "comparison_inventory.json").read_text())
    assert inventory["comparison_status"] == "complete" and len(inventory["comparisons"]) == 18
    assert len(pd.read_csv(tables / "paired_comparisons.csv")) == 18 * len(comparisons.METRICS)
    delong = pd.read_csv(tables / "delong_comparisons.csv")
    assert len(delong) == 18 and delong.status.eq("unsupported_design").all() and delong.p_value.isna().all()
    for name, value in (("RESULTS_DIR", results), ("TABLES_DIR", tables), ("FIGURES_DIR", paper / "figures"),
                        ("REPORTS_DIR", paper / "reports")):
        monkeypatch.setattr(make_report, name, value)
    _disable_legacy_exports(monkeypatch)
    monkeypatch.setattr(make_report, "_export_comparison_tables", lambda *args, **kwargs: None)
    rendered = []
    monkeypatch.setattr(decision_curve, "render_comparison_curves", lambda data, pairs, path, **kwargs:
                        rendered.append((data.copy(), pairs.copy(), kwargs)))
    monkeypatch.setattr(decision_curve, "net_benefit", lambda *args, **kwargs: pytest.fail("Saved report must not recalculate DCA"))
    monkeypatch.setattr(metrics, "summarize", lambda *args, **kwargs: pytest.fail("Saved report must not regenerate uncertainty"))
    monkeypatch.setattr(sys, "argv", ["make_report", "--strict", "--run-manifest", str(manifest)])
    make_report.main()
    pd.testing.assert_frame_equal(rendered[0][0], pd.read_csv(tables / "decision_curve_summary.csv"))
    pd.testing.assert_frame_equal(rendered[0][1], pd.read_csv(tables / "decision_curve_comparisons.csv"))
    assert rendered[0][2]["comparison_status"] == "complete"
    status = json.loads((paper / "reports/report_status.json").read_text())
    assert status["comparison_coverage"] == {"available": 18}
    assert status["comparison_status"] == "complete"
    assert status["delong_status_counts"] == {"unsupported_design": 18}
