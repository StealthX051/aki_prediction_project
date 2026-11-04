import json
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from reporting.display_dictionary import load_display_dictionary
from reporting.preop_descriptives import _build_descriptive_table, _ecg_display, _filter_to_report_cohort, _restrict_to_roster
from reporting.missingness_table import compute_missingness_table
from reporting.cohort_flow import flow_from_cohort, _stage_node_lines, normalize_counts
from reporting.feature_set_display import order_feature_sets
from results_recreation import results_analysis as analysis
from results_recreation import metrics_summary as metrics


def _population():
    return pd.DataFrame({"caseid": [1, 2, 3, 4], "subjectid": [10, 10, 20, 30],
        "eligible_any_aki": [1, 1, 0, 1], "aki_label": [0, 1, None, 0],
        "eligible_icu_admission": [1] * 4, "y_icu_admit": [0, 1, 1, 0],
        "ane_type": ["General"] * 4, "age": [40, 60, 90, None],
        "preop_pt": [94, 101, 109, None], "preop_ecg": ["Normal Sinus Rhythm", "Atrial fibrillation", None, "Unknown"]})


def test_table_one_sections_median_missingness_and_ecg_without_model_mutation():
    original = _population()
    cohort = _filter_to_report_cohort(original, "main_shared")
    copy = cohort.copy(deep=True)
    table = _build_descriptive_table(cohort, None, continuous_features=["age", "preop_pt"],
        categorical_features=["ane_type", "preop_ecg"], display_dictionary=load_display_dictionary(),
        alpha=.05, max_sample=5000, random_state=42)
    assert table.loc[table.Type.eq("Section"), "Feature"].tolist() == [
        "Demographics", "Surgical characteristics", "Clinical comorbidities and tests", "Laboratory values"]
    assert "Normality p-value" not in table
    assert table.loc[table.Feature.eq("Age (years)"), "Summary"].iloc[0] == "50.00 (45.00, 55.00)"
    assert "1 (33.3%)" in table["Missing n (%)"].tolist()
    assert table.loc[table.Type.eq("Population"), "Summary"].item() == "3 operations among 2 unique patients"
    assert set(_ecg_display(original.preop_ecg)) == {"Normal sinus rhythm", "Abnormal", "Missing", "Unknown"}
    pd.testing.assert_frame_equal(copy, cohort)


def test_roster_aligns_descriptive_missingness_and_flow_denominators(tmp_path):
    roster = tmp_path / "roster.csv"
    pd.DataFrame({"caseid": [1, 2]}).to_csv(roster, index=False)
    cohort = _restrict_to_roster(_filter_to_report_cohort(_population(), "main_shared"), roster)
    table = compute_missingness_table(cohort, load_display_dictionary(), exclude_columns=["caseid"])
    assert set(table.n_operations) == {2}
    assert set(table.n_patients) == {1}
    flow = flow_from_cohort(_population(), "main_shared", roster)
    assert flow.stages[-1].count == 2
    assert flow.stages[-1].n_patients == 1
    assert "1 unique patients" in _stage_node_lines(flow.stages[-1])
    assert len(_filter_to_report_cohort(_population(), "icu_waveform_sensitivity")) == 4


def test_flow_retains_patient_counts_and_zero_event_denominator():
    flow = normalize_counts({"total_cases": 5, "total_patients": 4,
        "final_cohort": {"count": 3, "n_patients": 2}, "label_split": {"true": 0, "false": 3}})
    assert flow.stages[-1].n_patients == 2
    assert flow.outcome_split is not None
    assert flow.outcome_split.true_count == 0


def test_prepared_flow_preserves_source_denominators_and_skips_noop_stages(tmp_path):
    metadata = {"outer_flow_counts": {"source_file": "/source/clinical.csv", "stages": [
        {"key": key, "title": title, "count": count, "n_patients": patients}
        for key, title, count, patients in [
            ("source", "All source operations", 8, 7),
            ("departments", "Selected surgical departments", 8, 7),
            ("opend", "Known operation end", 8, 7),
            ("general_anesthesia", "General anesthesia", 6, 5),
            ("asa", "Exclude ASA V/VI; retain missing ASA", 5, 4),
            ("waveforms", "Required waveform availability", 4, 3)]]}}
    flow = flow_from_cohort(_population(), "main_shared", artifact_metadata=metadata)
    assert [s.count for s in flow.stages] == [8, 6, 5, 4, 3]
    assert [s.n_patients for s in flow.stages] == [7, 5, 4, 3, 2]
    assert "Selected surgical departments" not in [s.title for s in flow.stages]
    assert [s.removal_reason for s in flow.stages[1:4]] == [
        "Non-general anesthesia", "ASA V/VI", "Required waveform unavailable"]
    assert all(s.removed is None or s.removed > 0 for s in flow.stages)
    subset = _population().iloc[:2]
    with pytest.raises(ValueError, match="diagnostic trim"):
        flow_from_cohort(subset, "main_shared", artifact_metadata=metadata)
    metadata["diagnostic_trim"] = {"before_operations": 4, "before_patients": 3,
                                    "after_operations": 2, "after_patients": 1}
    metadata["cohort_scopes"] = {"main_shared": {"operations": 3, "patients": 2}}
    flow = flow_from_cohort(subset, "main_shared", artifact_metadata=metadata)
    assert [s.count for s in flow.stages] == [8, 6, 5, 4, 3, 2]
    assert "Diagnostic subset (enriched; not a clinical exclusion)" in flow.stages[-1].detail
    assert "diagnostic subset" in flow.stages[-1].detail
    metadata["diagnostic_trim"]["cohort_scope"] = "icu_waveform_sensitivity"
    with pytest.raises(ValueError, match="different cohort scope"):
        flow_from_cohort(subset, "main_shared", artifact_metadata=metadata)
    metadata["diagnostic_trim"]["cohort_scope"] = "main_shared"
    metadata["outer_flow_counts"]["stages"][3]["count"] = 9
    with pytest.raises(ValueError, match="decrease"):
        flow_from_cohort(subset, "main_shared", artifact_metadata=metadata)


def test_table_one_population_counts_survive_each_export(tmp_path, monkeypatch):
    from docx import Document
    import reporting.preop_descriptives as descriptive
    table = descriptive._build_descriptive_table(_population(), None,
        continuous_features=["age"], categorical_features=[], display_dictionary=load_display_dictionary(),
        alpha=.05, max_sample=5000, random_state=42)
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr(descriptive, "TABLES_DIR", tmp_path)
    monkeypatch.setattr(descriptive, "refresh_paper_bundle", lambda root: None)
    pdf_sections = []
    def save_pdf(path, *, title, sections):
        pdf_sections.extend(sections)
        path.touch()
    monkeypatch.setattr(descriptive, "write_pdf_sections", save_pdf)
    descriptive._save_table(table, "population")
    note = "4 operations among 3 unique patients"
    for suffix in ("csv", "html", "md", "tex"):
        assert note in (tmp_path / f"population.{suffix}").read_text()
    assert any(note in paragraph.text for paragraph in Document(tmp_path / "population.docx").paragraphs)
    assert any(note in section.get("text", "") for section in pdf_sections)


def test_core_feature_order_and_pooled_curve_scores():
    frame = pd.DataFrame({"Feature Set Key": ["preop_and_all_waveforms", "all_waveforms", "preop_only"]})
    assert order_feature_sets(frame)["Feature Set Key"].tolist() == ["preop_only", "all_waveforms", "preop_and_all_waveforms"]
    predictions = pd.DataFrame({"y_true": [0, 1, 0, 1], "y_prob_calibrated": [.3, .4, .1, .9], "outer_fold_id": [0, 0, 1, 1]})
    roc = analysis._curve_stats(predictions, analysis.PlotConfig(), "roc")
    pr = analysis._curve_stats(predictions, analysis.PlotConfig(), "pr")
    assert roc["score"] == roc_auc_score(predictions.y_true, predictions.y_prob_calibrated)
    assert pr["score"] == average_precision_score(predictions.y_true, predictions.y_prob_calibrated)
    assert not roc["band"]


def _pred(feature="preop_only", family="XGBoost", scope="main_shared", probabilities=None):
    frame = pd.DataFrame({"caseid": [1, 2, 3, 4, 5, 6], "subjectid": [10, 10, 20, 30, 30, 40],
        "y_true": [0, 1, 0, 1, 0, 1], "y_prob_calibrated": probabilities or [.2, .6, .3, .7, .5, .6],
        "threshold": [.5] * 6, "outer_fold_id": [0, 0, 1, 1, 1, 0]})
    frame["y_pred_label"] = (frame.y_prob_calibrated >= .5).astype(int)
    return metrics.PredictionSet(Path(feature), frame, .5, "any_aki", "non_windowed", feature, family, "nested", scope, "nested_cv")


def test_paired_deltas_align_cases_and_match_hand_computed_cluster_draws():
    reference = _pred()
    candidate = _pred("all_waveforms", probabilities=[.1, .8, .2, .9, .3, .8])
    candidate.df = candidate.df.iloc[::-1].reset_index(drop=True)
    summary, bootstrap = metrics.summarize([reference, candidate], n_bootstrap=40,
        delta_mode="reference", n_jobs=1, parallel_backend="threads", stratified_bootstrap=True)
    target = summary.loc[summary.feature_set.eq("all_waveforms")].iloc[0]
    base = reference.df.sort_values("caseid").reset_index(drop=True)
    alternative = candidate.df.sort_values("caseid").reset_index(drop=True)
    differences = []
    for draw in metrics._generate_bootstrap_indices(base, 40, True, 0):
        differences.append(brier_score_loss(alternative.y_true.iloc[draw], alternative.y_prob_calibrated.iloc[draw])
                           - brier_score_loss(base.y_true.iloc[draw], base.y_prob_calibrated.iloc[draw]))
    assert target.delta_brier == pytest.approx(brier_score_loss(alternative.y_true, alternative.y_prob_calibrated) - brier_score_loss(base.y_true, base.y_prob_calibrated))
    assert target.delta_brier_lower == pytest.approx(np.percentile(differences, 2.5))
    assert target.delta_brier_upper == pytest.approx(np.percentile(differences, 97.5))


def test_single_class_bootstrap_draws_retain_defined_metrics_and_paired_counts(caplog):
    reference, candidate = _pred(), _pred("all_waveforms")
    reference.df = pd.DataFrame({
        "caseid": [1, 2], "subjectid": [10, 20], "y_true": [0, 1],
        "y_prob_calibrated": [.1, .8], "y_pred_label": [0, 1],
        "threshold": [.5, .5], "outer_fold_id": [0, 1],
    })
    candidate.df = reference.df.assign(y_prob_calibrated=[.2, .7])
    draws = metrics._generate_bootstrap_indices(reference.df, 128, False, 0)
    summary, samples = metrics.summarize(
        [reference, candidate], n_bootstrap=128, stratified_bootstrap=False,
        delta_mode="reference", n_jobs=1, parallel_backend="threads", strict=True,
    )
    target = summary.loc[summary.feature_set.eq("all_waveforms")].iloc[0]
    assert target.bootstrap_replicates == 128
    assert len(samples.loc[samples.feature_set.eq("all_waveforms")]) == 128
    has_both = [reference.df.y_true.iloc[draw].nunique() == 2 for draw in draws]
    has_positive = [reference.df.y_true.iloc[draw].eq(1).any() for draw in draws]
    assert 0 < sum(has_both) < 128
    assert target.auroc_bootstrap_valid == target.delta_auroc_bootstrap_valid == sum(has_both)
    assert target.auprc_bootstrap_valid == target.delta_auprc_bootstrap_valid == sum(has_positive)
    assert target.brier_bootstrap_valid == target.delta_brier_bootstrap_valid == 128
    candidate_brier, paired_brier = [], []
    for draw in draws:
        truth = reference.df.y_true.to_numpy()[draw]
        base = np.mean((reference.df.y_prob_calibrated.to_numpy()[draw] - truth) ** 2)
        alternative = np.mean((candidate.df.y_prob_calibrated.to_numpy()[draw] - truth) ** 2)
        candidate_brier.append(alternative)
        paired_brier.append(alternative - base)
    np.testing.assert_allclose([target.brier_lower, target.brier_upper], np.percentile(candidate_brier, [2.5, 97.5]))
    np.testing.assert_allclose([target.delta_brier_lower, target.delta_brier_upper], np.percentile(paired_brier, [2.5, 97.5]))
    assert "auroc" in caplog.text and "finite replicates" in caplog.text


@pytest.mark.parametrize("truth", [0, 1])
def test_single_class_metric_definitions_preserve_confusion_matrix_dimensions(truth):
    result = metrics._compute_metrics_arrays(
        np.array([truth, truth]), np.array([.1, .2]), np.array([truth, truth]), .5,
    )
    assert np.isnan(result["auroc"])
    assert result["accuracy"] == 1
    assert result["brier"] == pytest.approx(np.mean((np.array([.1, .2]) - truth) ** 2))
    if truth:
        assert result["auprc"] == result["sensitivity"] == 1
        assert np.isnan(result["specificity"])
    else:
        assert np.isnan(result["auprc"]) and np.isnan(result["sensitivity"])
        assert result["specificity"] == 1
        assert result["precision"] == result["f1"] == 0


def test_insufficient_defined_bootstrap_draws_do_not_create_an_interval(caplog):
    lower, upper, count = metrics._bootstrap_interval([np.nan, .5], 2, label="test/auroc")
    assert count == 1 and np.isnan(lower) and np.isnan(upper)
    assert "1/2 finite replicates" in caplog.text


def test_strict_reference_override_cannot_change_model_family(tmp_path):
    reference, target = _pred(family="EBM"), _pred("all_waveforms")
    with pytest.raises(metrics.ValidationError, match="same-family reference"):
        metrics.summarize([reference, target], n_bootstrap=0, strict=True,
                          delta_mode="reference", reference_model_name="EBM")
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({
        "cohort_scope": "main_shared", "validation_scheme": "nested_cv", "diagnostic": True,
        "configurations": [{"model_type": "xgboost", "outcome": "any_aki",
                            "branch": "non_windowed", "feature_set": "all_waveforms"}],
    }))
    with pytest.raises(metrics.ValidationError, match="same-family reference"):
        metrics._validate_run_manifest([reference, target], manifest, require_reference=True,
                                       reference_model_name="EBM")
    legacy, _ = metrics.summarize([reference, target], n_bootstrap=0,
                                 delta_mode="reference", reference_model_name="EBM")
    assert np.isfinite(legacy.loc[legacy.feature_set.eq("all_waveforms"), "delta_brier"]).all()


@pytest.mark.parametrize("previous_version", [None, 1])
def test_strict_report_requires_current_uncertainty_protocol(previous_version, monkeypatch):
    import sys
    from reporting import make_report

    candidate = _pred()
    summary, _ = metrics.summarize([candidate], n_bootstrap=20, n_jobs=1,
                                   parallel_backend="threads", strict=True)
    assert summary.uncertainty_protocol_version.eq(metrics.UNCERTAINTY_PROTOCOL_VERSION).all()
    if previous_version is None:
        summary = summary.drop(columns="uncertainty_protocol_version")
    else:
        summary["uncertainty_protocol_version"] = previous_version
    monkeypatch.setattr(make_report, "crawl_predictions", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(make_report, "_load_metrics", lambda *args: summary)
    monkeypatch.setattr(make_report, "prepare_metrics_for_display", lambda *args: pytest.fail("Must reject before export"))
    monkeypatch.setattr(sys, "argv", ["make_report", "--strict"])
    with pytest.raises(ValueError, match="stale uncertainty protocol"):
        make_report.main()


def test_strict_report_rejects_stale_intervals_when_point_metrics_still_match(monkeypatch):
    import sys
    from reporting import make_report

    candidate = _pred()
    summary, _ = metrics.summarize([candidate], n_bootstrap=20, bootstrap_seed=17,
        delta_mode="reference", n_jobs=1, parallel_backend="threads", strict=True, stratified_bootstrap=True)
    assert summary.loc[0, "bootstrap_replicates"] == 20
    assert summary.loc[0, "bootstrap_seed"] == 17
    assert bool(summary.loc[0, "bootstrap_stratified"])
    assert summary.loc[0, "reference_feature_set"] == "preop_only"
    assert summary.loc[0, "reference_model_name"] == "XGBoost"
    assert summary.loc[0, "delta_mode"] == "reference"
    original_hash = metrics._prediction_content_hash(candidate)
    candidate.df = candidate.df.iloc[::-1].reset_index(drop=True)
    assert metrics._prediction_content_hash(candidate) == original_hash
    original_metrics = metrics.compute_point_metrics(candidate)
    # Permuting scores among negative patients preserves all point metrics but
    # changes their association with the patient clusters used for uncertainty.
    for caseid, probability in ((1, .3), (3, .2)):
        candidate.df.loc[candidate.df.caseid.eq(caseid), "y_prob_calibrated"] = probability
    assert metrics.compute_point_metrics(candidate) == original_metrics
    assert metrics._prediction_content_hash(candidate) != original_hash
    monkeypatch.setattr(make_report, "crawl_predictions", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(make_report, "_load_metrics", lambda *args: summary)
    def forbid_exports(*args):
        pytest.fail("Stale uncertainty must be rejected before report exports")
    monkeypatch.setattr(make_report, "prepare_metrics_for_display", forbid_exports)
    monkeypatch.setattr(sys, "argv", ["make_report", "--strict"])
    with pytest.raises(ValueError, match="prediction/protocol provenance"):
        make_report.main()


def test_summary_digest_binds_saved_validation_fingerprint(tmp_path):
    candidate, _ = _saved_protocol(_pred(), tmp_path)
    original_hash = metrics._prediction_content_hash(candidate)
    path = candidate.path.parent.parent / "artifacts" / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["validation_fingerprint"]["code_hash"] = "new_model_protocol"
    path.write_text(json.dumps(metadata))
    assert metrics._prediction_content_hash(candidate) != original_hash


def test_strict_report_rejects_target_delta_from_previous_reference(monkeypatch):
    import sys
    from reporting import make_report

    reference, target = _pred(), _pred("all_waveforms", probabilities=[.1, .8, .2, .9, .3, .8])
    old_summary, _ = metrics.summarize([reference, target], n_bootstrap=20,
        delta_mode="reference", n_jobs=1, parallel_backend="threads", strict=True)
    reference.df.loc[reference.df.caseid.eq(1), "y_prob_calibrated"] = .3
    reference.df.loc[reference.df.caseid.eq(3), "y_prob_calibrated"] = .2
    current_reference_summary, _ = metrics.summarize([reference], n_bootstrap=20,
        delta_mode="reference", n_jobs=1, parallel_backend="threads", strict=True)
    # A fresh reference row must not legitimize a target's paired interval
    # computed against the reference's previous patient/probability association.
    mixed_summary = pd.concat([current_reference_summary,
        old_summary.loc[old_summary.feature_set.eq("all_waveforms")]], ignore_index=True)
    monkeypatch.setattr(make_report, "crawl_predictions", lambda *args, **kwargs: [reference, target])
    monkeypatch.setattr(make_report, "_load_metrics", lambda *args: mixed_summary)
    def forbid_exports(*args):
        pytest.fail("Stale paired uncertainty must be rejected before report exports")
    monkeypatch.setattr(make_report, "prepare_metrics_for_display", forbid_exports)
    monkeypatch.setattr(sys, "argv", ["make_report", "--strict"])
    with pytest.raises(ValueError, match="reference provenance"):
        make_report.main()


def test_summary_digest_survives_csv_formatting_and_summary_roundtrip(tmp_path, monkeypatch):
    import sys
    from reporting import make_report

    candidate, _ = _saved_protocol(_pred(), tmp_path)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    candidate.df.to_csv(first, index=False)
    candidate.df.iloc[::-1, ::-1].to_csv(second, index=False, float_format="%.12f")
    assert first.read_bytes() != second.read_bytes()
    candidate.df = pd.read_csv(first)
    original_hash = metrics._prediction_content_hash(candidate)
    summary, _ = metrics.summarize([candidate], n_bootstrap=0, delta_mode="reference", strict=True)
    saved_summary = tmp_path / "metrics_summary.csv"
    summary.to_csv(saved_summary, index=False)
    loaded_summary = pd.read_csv(saved_summary)
    candidate.df = pd.read_csv(second)
    assert metrics._prediction_content_hash(candidate) == original_hash
    assert loaded_summary.loc[0, "prediction_content_hash"] == original_hash
    metadata_path = candidate.path.parent.parent / "artifacts" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_path.write_text(json.dumps(metadata, sort_keys=True, indent=2))
    assert metrics._prediction_content_hash(candidate) == original_hash
    monkeypatch.setattr(make_report, "crawl_predictions", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(make_report, "_load_metrics", lambda *args: loaded_summary)
    monkeypatch.setattr(make_report, "_load_comparisons", lambda *args, **kwargs: {})
    def stop_before_exports(*args):
        raise RuntimeError("fresh CSV summary reached exports")
    monkeypatch.setattr(make_report, "prepare_metrics_for_display", stop_before_exports)
    monkeypatch.setattr(sys, "argv", ["make_report", "--strict"])
    with pytest.raises(RuntimeError, match="fresh CSV summary reached exports"):
        make_report.main()


def test_no_cross_family_or_cross_cohort_reference_fallback():
    summary, _ = metrics.summarize([_pred(), _pred("all_waveforms", family="EBM"),
        _pred("all_waveforms", scope="icu_waveform_sensitivity")],
        n_bootstrap=10, delta_mode="reference", n_jobs=1, parallel_backend="threads")
    targets = summary.loc[summary.feature_set.eq("all_waveforms")]
    assert targets.delta_brier.isna().all()


def test_strict_requires_manifest_configurations_and_patient_ids(tmp_path):
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"cohort_scope": "main_shared", "validation_scheme": "nested_cv",
        "diagnostic": True, "configurations": [{"model_type": "xgboost", "outcome": "any_aki", "branch": "non_windowed", "feature_set": "all_waveforms"}]}))
    with pytest.raises(metrics.ValidationError, match="Missing required"):
        metrics._validate_run_manifest([_pred()], manifest)
    manifest_data = json.loads(manifest.read_text())
    manifest_data["configurations"][0]["feature_set"] = "preop_only"
    from data_preparation.cohort_scopes import selected_case_hash
    manifest_data["branch_rosters"] = {"non_windowed": {"selected_case_hash": selected_case_hash(_pred().df)}}
    manifest.write_text(json.dumps(manifest_data))
    saved, _ = _saved_protocol(_pred(), tmp_path)
    loaded = metrics._validate_run_manifest([saved], manifest)
    assert loaded[0].analysis_status == "diagnostic"
    missing = _pred()
    missing.df = missing.df.drop(columns="subjectid")
    with pytest.raises(metrics.ValidationError, match="patient"):
        metrics.summarize([missing], strict=True)


@pytest.mark.parametrize("probabilities", [[.1, .95, .4, .8], [.25] * 4])
def test_calibration_main_panel_displays_high_risk_and_constant_predictions(monkeypatch, tmp_path, probabilities):
    import matplotlib.pyplot as plt
    panels = {}
    def save(fig, path, **kwargs):
        panels[Path(path).stem] = (fig.axes[0].get_xlim(), fig.axes[0].lines[0].get_xydata())
        plt.close(fig)
        return []
    monkeypatch.setattr(analysis, "save_report_figure_bundle", save)
    monkeypatch.setattr(analysis, "refresh_paper_bundle", lambda root: None)
    frame = pd.DataFrame({"caseid": [1, 2, 3, 4], "outcome": ["icu_admission"] * 4,
        "branch": ["non_windowed"] * 4, "model_name": ["EBM"] * 4,
        "feature_set": ["preop_only"] * 4, "y_true": [0, 1, 0, 1],
        "y_prob_calibrated": probabilities})
    analysis.plot_curves(frame, analysis.PlotConfig(n_jobs=1, show_prob_hist=False))
    limits, points = panels["calibration_icu_admission_non_windowed_EBM"]
    assert limits == (0.0, 1.0)
    assert len(points) > 0
    if len(set(probabilities)) == 1:
        np.testing.assert_allclose(points, [[.25, .5]])


@pytest.mark.parametrize("status", ["diagnostic", "definitive"])
def test_decision_curve_visible_status_matches_saved_metadata(tmp_path, monkeypatch, status):
    import matplotlib.pyplot as plt
    import reporting.decision_curve as decision
    frame = pd.DataFrame({"caseid": [1, 2, 3, 4], "subjectid": [10, 20, 30, 40],
        "outcome": "any_aki", "branch": "non_windowed", "model_name": "EBM",
        "feature_set": "preop_only", "y_true": [0, 1, 0, 1],
        "y_prob_calibrated": [.1, .7, .2, .5], "analysis_status": status})
    titles = []
    def save(fig, *args, **kwargs):
        titles.append(fig.axes[0].get_title())
        plt.close(fig)
        return []
    monkeypatch.setattr(decision, "save_report_figure_bundle", save)
    decision.export_decision_curves(frame, tmp_path)
    assert titles and all(f"[{status}]" in title for title in titles)
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["analysis_status"] == status


def test_point_deltas_are_available_without_bootstrap():
    summary, samples = metrics.summarize([_pred(), _pred("all_waveforms", probabilities=[.1, .8, .2, .9, .3, .8])], n_bootstrap=0, delta_mode="reference")
    assert samples is None
    assert summary.loc[summary.feature_set.eq("all_waveforms"), "delta_brier"].iloc[0] < 0


def test_strict_rejects_incomplete_frozen_roster_and_declared_hash(tmp_path):
    from data_preparation.cohort_scopes import selected_case_hash
    candidate = _pred()
    manifest = {"cohort_scope": "main_shared", "validation_scheme": "nested_cv", "diagnostic": True,
        "configurations": [{"model_type": "xgboost", "outcome": "any_aki", "branch": "non_windowed", "feature_set": "preop_only"}],
        "branch_rosters": {"non_windowed": {"selected_case_hash": selected_case_hash(candidate.df)}}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    candidate.df = candidate.df.iloc[:-1]
    with pytest.raises(metrics.ValidationError, match="frozen branch roster"):
        metrics._validate_run_manifest([candidate], manifest_path)
    candidate = _pred()
    candidate.df["selected_case_hash"] = "incorrect"
    with pytest.raises(metrics.ValidationError, match="selected_case_hash"):
        metrics._validate_run_manifest([candidate], manifest_path)


@pytest.mark.parametrize("diagnostic", [False, True])
def test_lr_fold_explanations_use_saved_models_and_only_heldout_cases(tmp_path, monkeypatch, diagnostic):
    from sklearn.linear_model import LogisticRegression
    from model_creation.preprocessing import FoldPreprocessor
    from reporting.fold_explanations import export_fold_bundle
    import reporting.fold_explanations as explanations
    raw = pd.DataFrame({"caseid": [1, 2, 3, 4], "age": [40, 60, None, 80], "sex": ["M", "F", "M", "F"]})
    prep = FoldPreprocessor(impute_missing=True, scale_numeric=True, add_missing_indicators=True)
    # Nominally held-out rows must not participate in this fixture's training.
    train = prep.fit_transform(raw.loc[:1, ["age", "sex"]])
    model = LogisticRegression().fit(train, [0, 1])
    fold = tmp_path / "fold"
    fold.mkdir()
    raw.attrs["source_dataset_sha256"] = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    (fold / "metadata.json").write_text(json.dumps({"model_type": "logistic_regression",
        "validation_fingerprint": {"dataset_hash": raw.attrs["source_dataset_sha256"], "diagnostic": diagnostic}}))
    pd.DataFrame({"caseid": [4, 3], "y_true": [1, 0]}).to_csv(fold / "test_predictions.csv", index=False)
    with (fold / "model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    with (fold / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(prep, handle)
    monkeypatch.setattr(LogisticRegression, "fit", lambda *a, **kw: pytest.fail("Report must not fit a model"))
    monkeypatch.setattr(FoldPreprocessor, "fit", lambda *a, **kw: pytest.fail("Report must not fit preprocessing"))
    titles = []
    monkeypatch.setattr(explanations, "_bar", lambda *a, **kw: titles.append(a[3]) or [])
    destination = tmp_path / "explanations"
    export_fold_bundle(fold, raw, destination)
    coefficients = pd.read_csv(destination / "coefficients.csv")
    assert coefficients.feature.tolist() == prep.feature_names_
    np.testing.assert_allclose(coefficients.coefficient, model.coef_[0], rtol=1e-12)
    assert coefficients.loc[coefficients.feature.eq("age"), "interpretation"].iloc[0] == "per training SD"
    metadata = json.loads((destination / "explanation_metadata.json").read_text())
    assert metadata["caseids"] == [4, 3]
    assert metadata["heldout_operations"] == 2
    assert coefficients.loc[coefficients.feature.eq("age"), "display_label"].item() == "Age (years)"
    assert titles == [("Diagnostic: " if diagnostic else "") + "Outer-fold logistic regression coefficients"]


def test_saved_fold_explanations_reject_changed_features_before_loading(tmp_path, monkeypatch):
    import reporting.fold_explanations as explanations
    source = tmp_path / "modeling.csv"
    raw = pd.DataFrame({"caseid": [1, 2], "subjectid": [10, 20], "age": [40, 60]})
    raw.to_csv(source, index=False)
    fitted_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    fold = tmp_path / "results/models/lr/artifacts/folds/repeat_00/outer_00"
    fold.mkdir(parents=True)
    (fold / "metadata.json").write_text(json.dumps({"model_type": "logistic_regression",
        "validation_fingerprint": {"branch": "non_windowed", "dataset_hash": fitted_hash, "diagnostic": True}}))
    monkeypatch.setattr(explanations.utils, "FULL_FEATURES_FILE", source)
    monkeypatch.setattr(explanations.utils, "load_data", lambda *a: pytest.fail("Hash mismatch must precede loading"))
    raw.loc[0, "age"] = 80  # Patient and operation identity are unchanged.
    raw.to_csv(source, index=False)
    destination = tmp_path / "explanations"
    with pytest.raises(ValueError, match="dataset hash"):
        explanations.export_fold_explanations(tmp_path / "results", destination)
    assert not destination.exists()
    with pytest.raises(ValueError, match="provenance is missing"):
        explanations.export_fold_bundle(fold, raw, destination)


def test_fold_explanations_reject_incomplete_new_integrity_protocol(tmp_path):
    from reporting.fold_explanations import export_fold_bundle
    raw = pd.DataFrame({"caseid": [1], "age": [40]})
    raw.attrs["source_dataset_sha256"] = "sourcehash"
    (tmp_path / "metadata.json").write_text(json.dumps({"model_type": "logistic_regression",
        "checkpoint_bundle_protocol_version": 2,
        "validation_fingerprint": {"dataset_hash": "sourcehash", "diagnostic": True,
                                   "checkpoint_bundle_protocol_version": 2}}))
    with pytest.raises(ValueError, match="complete verified outer-fold bundle"):
        export_fold_bundle(tmp_path, raw, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def _saved_protocol(ps, tmp_path, *, diagnostic=True, outer_folds=2, inner_folds=2, calibration_folds=5):
    model_dir = tmp_path / ps.model_name / ps.feature_set
    ps.path = model_dir / "predictions" / "test.csv"
    ps.path.parent.mkdir(parents=True, exist_ok=True)
    ps.df["diagnostic"] = diagnostic
    artifact_dir = model_dir / "artifacts"
    artifact_dir.mkdir()
    fingerprint = {"validation_protocol_version": 4, "code_revision": "revision", "code_hash": "code",
        "environment": {"numpy": "pinned"}, "preprocessing_protocol": "fold-local",
        "compute_profile": "manuscript",
        "dataset_hash": "frozen_modeling_csv",
        "diagnostic": diagnostic, "cohort_scope": ps.cohort_scope, "validation_scheme": ps.validation_scheme,
        "outer_folds": outer_folds, "inner_folds": inner_folds, "calibration_folds": calibration_folds, "repeats": 1}
    metadata = {"diagnostic": diagnostic, "validation_fingerprint": fingerprint,
                "compute_profile": "manuscript",
                "outer_folds": outer_folds, "inner_folds": inner_folds, "calibration_folds": calibration_folds, "repeats": 1}
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata))
    return ps, {key: fingerprint[key] for key in ("validation_protocol_version", "code_revision", "code_hash", "environment", "preprocessing_protocol")}


def test_supplement_manifest_uses_cached_same_family_reference(tmp_path):
    from data_preparation.cohort_scopes import selected_case_hash
    reference, identity = _saved_protocol(_pred(), tmp_path)
    target, _ = _saved_protocol(_pred("awp_only"), tmp_path)
    manifest = {"cohort_scope": "main_shared", "validation_scheme": "nested_cv", "diagnostic": True,
        "study_identity": identity,
        "configurations": [{"model_type": "xgboost", "outcome": "any_aki", "branch": "non_windowed", "feature_set": "awp_only"}],
        "branch_rosters": {"non_windowed": {"selected_case_hash": selected_case_hash(target.df), "dataset_sha256": "frozen_modeling_csv"}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    loaded = metrics._validate_run_manifest([reference, target], path, require_reference=True)
    assert {ps.feature_set for ps in loaded} == {"awp_only", "preop_only"}
    with pytest.raises(metrics.ValidationError, match="reference dependency"):
        metrics._validate_run_manifest([target], path, require_reference=True)


def test_strict_rejects_mixed_code_and_promoted_reduced_validation(tmp_path):
    from data_preparation.cohort_scopes import selected_case_hash
    reference, identity = _saved_protocol(_pred(), tmp_path)
    target, _ = _saved_protocol(_pred("all_waveforms"), tmp_path)
    manifest = {"cohort_scope": "main_shared", "validation_scheme": "nested_cv", "diagnostic": True,
        "study_identity": identity,
        "configurations": [{"model_type": "xgboost", "outcome": "any_aki", "branch": "non_windowed", "feature_set": "all_waveforms"}],
        "branch_rosters": {"non_windowed": {"selected_case_hash": selected_case_hash(target.df), "dataset_sha256": "frozen_modeling_csv"}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    metadata_path = target.path.parent.parent / "artifacts" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["validation_fingerprint"]["code_hash"] = "stale"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(metrics.ValidationError, match="code/environment"):
        metrics._validate_run_manifest([reference, target], path)
    metadata["validation_fingerprint"]["code_hash"] = "code"
    metadata["validation_fingerprint"]["dataset_hash"] = "same_cases_changed_features"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(metrics.ValidationError, match="frozen branch dataset hash"):
        metrics._validate_run_manifest([reference, target], path)
    metadata["validation_fingerprint"]["dataset_hash"] = "frozen_modeling_csv"
    metadata_path.write_text(json.dumps(metadata))
    manifest["diagnostic"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(metrics.ValidationError, match="Diagnostic status"):
        metrics._validate_run_manifest([reference, target], path)
    for ps in (reference, target):
        ps.df["diagnostic"] = False
        meta_path = ps.path.parent.parent / "artifacts" / "metadata.json"
        data = json.loads(meta_path.read_text())
        data["diagnostic"] = False
        data["validation_fingerprint"]["diagnostic"] = False
        meta_path.write_text(json.dumps(data))
    with pytest.raises(metrics.ValidationError, match="5 outer / 3 HPO / 5 calibration"):
        metrics._validate_run_manifest([reference, target], path)


@pytest.mark.parametrize("location,field,value", [
    ("metadata", "inner_folds", 5), ("metadata", "calibration_folds", 3),
    ("fingerprint", "inner_folds", 5), ("fingerprint", "calibration_folds", 3),
    ("fold", "actual_inner_folds", 2), ("fold", "actual_calibration_folds", 3),
    ("fold", "requested_calibration_folds", 3),
])
def test_definitive_report_requires_exact_five_outer_three_hpo_five_calibration(tmp_path, monkeypatch, location, field, value):
    from data_preparation.cohort_scopes import selected_case_hash
    import model_creation.validation as validation
    candidate, identity = _saved_protocol(_pred(), tmp_path, diagnostic=False,
                                          outer_folds=5, inner_folds=3, calibration_folds=5)
    candidate.df["outer_fold_id"] = [0, 1, 2, 3, 4, 0]
    candidate.df["repeat_id"] = 0
    artifact_dir = candidate.path.parent.parent / "artifacts"
    root_path = artifact_dir / "metadata.json"
    root_metadata = json.loads(root_path.read_text())
    fold_paths = []
    for fold_id in range(5):
        fold_path = artifact_dir / "folds" / "repeat_00" / f"outer_{fold_id:02d}" / "metadata.json"
        fold_path.parent.mkdir(parents=True)
        fold_path.write_text(json.dumps({"status": "completed", "actual_inner_folds": 3,
            "requested_inner_folds": 3, "actual_calibration_folds": 5, "requested_calibration_folds": 5,
            "validation_fingerprint": root_metadata["validation_fingerprint"]}))
        fold_paths.append(fold_path)
    # Bundle-byte verification and union alignment have dedicated adversarial
    # tests. This fixture isolates the definitive protocol's fold requirements.
    monkeypatch.setattr(validation, "is_outer_fold_checkpoint_complete", lambda path: True)
    monkeypatch.setattr(metrics, "_validate_nested_prediction_union", lambda *args: None)
    manifest = {"cohort_scope": "main_shared", "validation_scheme": "nested_cv", "diagnostic": False,
        "study_identity": identity,
        "configurations": [{"model_type": "xgboost", "outcome": "any_aki", "branch": "non_windowed", "feature_set": "preop_only"}],
        "branch_rosters": {"non_windowed": {"selected_case_hash": selected_case_hash(candidate.df), "dataset_sha256": "frozen_modeling_csv"}}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    assert metrics._validate_run_manifest([candidate], manifest_path)[0].analysis_status == "definitive"
    if location == "fold":
        changed = json.loads(fold_paths[0].read_text())
        changed[field] = value
        fold_paths[0].write_text(json.dumps(changed))
    else:
        target = root_metadata if location == "metadata" else root_metadata["validation_fingerprint"]
        target[field] = value
        root_path.write_text(json.dumps(root_metadata))
    with pytest.raises(metrics.ValidationError, match="5 outer / 3 HPO / 5 calibration|Incomplete or reduced"):
        metrics._validate_run_manifest([candidate], manifest_path)


def test_strict_nested_root_predictions_must_equal_verified_fold_union(tmp_path):
    from data_preparation.cohort_scopes import selected_case_hash
    from model_creation.validation import is_outer_fold_checkpoint_complete
    candidate, identity = _saved_protocol(_pred(), tmp_path)
    candidate.df["repeat_id"] = 0
    candidate.df["y_prob_raw"] = candidate.df.y_prob_calibrated
    artifact_dir = candidate.path.parent.parent / "artifacts"
    root_metadata = json.loads((artifact_dir / "metadata.json").read_text())
    fingerprint = root_metadata["validation_fingerprint"]
    fingerprint["checkpoint_bundle_protocol_version"] = 2
    (artifact_dir / "metadata.json").write_text(json.dumps(root_metadata))
    for fold_id, frame in candidate.df.groupby("outer_fold_id"):
        fold = artifact_dir / "folds/repeat_00" / f"outer_{fold_id:02d}"
        fold.mkdir(parents=True)
        frame.to_csv(fold / "test_predictions.csv", index=False)
        candidate.df.loc[~candidate.df.caseid.isin(frame.caseid)].to_csv(fold / "train_oof.csv", index=False)
        for name in ("model.json", "preprocessor.pkl", "calibration.json", "threshold.json", "best_params.json"):
            (fold / name).write_text("{}")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in fold.iterdir()}
        metadata = {"status": "completed", "model_type": "xgboost", "model_path": "model.json",
            "checkpoint_bundle_protocol_version": 2, "validation_fingerprint": fingerprint,
            "model_sha256": hashes["model.json"], "preprocessor_sha256": hashes["preprocessor.pkl"],
            "artifact_sha256": hashes, "repeat_id": 0, "outer_fold_id": int(fold_id)}
        (fold / "metadata.json").write_text(json.dumps(metadata))
        assert is_outer_fold_checkpoint_complete(fold)
    manifest = {"cohort_scope": "main_shared", "validation_scheme": "nested_cv", "diagnostic": True,
        "study_identity": identity,
        "configurations": [{"model_type": "xgboost", "outcome": "any_aki", "branch": "non_windowed", "feature_set": "preop_only"}],
        "branch_rosters": {"non_windowed": {"selected_case_hash": selected_case_hash(candidate.df), "dataset_sha256": "frozen_modeling_csv"}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert metrics._validate_run_manifest([candidate], path)
    candidate.df = candidate.df.iloc[::-1].reset_index(drop=True)
    assert metrics._validate_run_manifest([candidate], path)  # Row order has no scientific meaning.
    index = candidate.df.index[candidate.df.caseid.eq(1)][0]
    candidate.df.loc[index, "y_prob_calibrated"] = .8
    candidate.df.loc[index, "y_pred_label"] = 1  # Coherent edited scores/labels remain invalid.
    with pytest.raises(metrics.ValidationError, match="verified outer-fold prediction union"):
        metrics._validate_run_manifest([candidate], path)


@pytest.mark.parametrize("requested,expected", [(-1, 3), (20, 3), (1, 1)])
def test_curve_workers_respect_available_cpus_and_actual_groups(monkeypatch, requested, expected):
    workers = []
    monkeypatch.setattr(analysis, "available_cpu_count", lambda: 4)
    monkeypatch.setattr(analysis, "refresh_paper_bundle", lambda root: None)
    class CaptureParallel:
        def __init__(self, n_jobs):
            workers.append(n_jobs)
        def __call__(self, tasks):
            list(tasks)  # Inspect dispatch without rendering or starting workers.
    monkeypatch.setattr(analysis, "Parallel", CaptureParallel)
    frame = pd.DataFrame({"outcome": "any_aki", "branch": "non_windowed",
        "model_name": ["EBM", "XGBoost", "Logistic regression"], "feature_set": "preop_only",
        "y_true": [0, 1, 0], "y_prob_calibrated": [.1, .8, .2]})
    analysis.plot_curves(frame, analysis.PlotConfig(n_jobs=requested))
    assert workers == [expected]


def test_table_one_keeps_derived_labs_without_replacing_source_values(monkeypatch):
    import argparse
    import reporting.preop_descriptives as descriptive
    source = _population()
    processed = source.assign(age=999, preop_crp=[1., 2., 3., 4.])
    args = argparse.Namespace(report_cohort=None, cohort_scope="main_shared", dataset=Path("cohort.csv"),
        processed_dataset=Path("preop.csv"), branch_roster=None, display_dictionary=None,
        alpha=.05, max_normality_sample=5000, random_state=42, output_prefix="table1")
    monkeypatch.setattr(descriptive, "parse_args", lambda: args)
    monkeypatch.setattr(descriptive, "_load_preop_data", lambda path: source.copy())
    monkeypatch.setattr(descriptive, "_load_optional_processed_data", lambda path: processed.copy())
    monkeypatch.setattr(descriptive, "_filter_and_validate_features", lambda *a, **kw: (["age", "preop_crp"], ["preop_ecg"]))
    saved = []
    monkeypatch.setattr(descriptive, "_save_table", lambda frame, prefix: saved.append(frame))
    descriptive.main()
    table = saved[0]
    assert table.loc[table.Characteristic.eq("Age (years)"), "Finding (N=3)"].iloc[0] == "50.00 (45.00, 55.00)"
    assert table.loc[table.Characteristic.eq("CRP (mg/dL)"), "Finding (N=3)"].iloc[0] == "2.00 (1.50, 3.00)"


@pytest.mark.parametrize("family", ["xgboost", "ebm"])
def test_native_fold_explanations_regenerate_without_fitting(family, tmp_path, monkeypatch):
    from model_creation.preprocessing import FoldPreprocessor
    import reporting.fold_explanations as explanations
    import matplotlib.pyplot as plt
    raw = pd.DataFrame({"caseid": np.arange(8), "age": [40, 60, 42, 70, 50, 80, 44, 72],
                        "sex": ["M", "F"] * 4})
    prep = FoldPreprocessor()
    prep.fit(raw.loc[:5, ["age", "sex"]])
    transformed = prep.transform(raw[["age", "sex"]])
    if family == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=3, max_depth=1, min_child_weight=0, n_jobs=1).fit(transformed.iloc[:6], [0, 1] * 3)
    else:
        from interpret.glassbox import ExplainableBoostingClassifier
        model = ExplainableBoostingClassifier(outer_bags=1, inner_bags=0, max_rounds=10,
                    validation_size=0, interactions=0, n_jobs=1, min_samples_leaf=1, max_bins=8).fit(transformed.iloc[:6], [0, 1] * 3)
        model.term_scores_[0][0] = -.75  # A saved missing-value effect must not disappear from exports.
    fold = tmp_path / "fold"
    fold.mkdir()
    raw.attrs["source_dataset_sha256"] = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    (fold / "metadata.json").write_text(json.dumps({"model_type": family,
        "validation_fingerprint": {"dataset_hash": raw.attrs["source_dataset_sha256"], "diagnostic": True}}))
    pd.DataFrame({"caseid": [6, 7], "y_true": [0, 1]}).to_csv(fold / "test_predictions.csv", index=False)
    if family == "xgboost":
        model.save_model(fold / "model.json")
    else:
        with (fold / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle)
    with (fold / "preprocessor.pkl").open("wb") as handle:
        pickle.dump(prep, handle)
    monkeypatch.setattr(type(model), "fit", lambda *a, **kw: pytest.fail("Explanation must not refit"))
    monkeypatch.setattr(FoldPreprocessor, "fit", lambda *a, **kw: pytest.fail("Explanation must not fit preprocessing"))
    titles = []
    monkeypatch.setattr(explanations, "_bar", lambda *a, **kw: titles.append(a[3]) or [])
    def no_disk_figures(fig, *a, **kw):
        if fig._suptitle is not None:
            titles.append(fig._suptitle.get_text())
        else:
            titles.extend(ax.get_title() for ax in fig.axes if ax.get_title())
        plt.close(fig)
        return []
    monkeypatch.setattr(explanations, "save_report_figure_bundle", no_disk_figures)
    destination = tmp_path / "explanations"
    explanations.export_fold_bundle(fold, raw, destination)
    expected = "heldout_shap_values.csv" if family == "xgboost" else "heldout_term_contributions.csv"
    assert pd.read_csv(destination / expected).caseid.tolist() == [6, 7]
    if family == "ebm":
        native = json.loads((destination / "native_explanation.json").read_text())
        assert native["terms"]
        special = pd.read_csv(destination / "special_bin_scores.csv")
        assert set(special.bin_type) == {"missing", "unknown"}
        assert special.loc[special.term.eq("age") & special.bin_type.eq("missing"), "raw_log_odds_contribution"].item() == -.75
        assert native["terms"][0]["special_bins"][0]["raw_log_odds_contribution"] == -.75
        assert "not causal effects" in native["interpretation"]
        contributions = pd.read_csv(destination / expected).drop(columns="caseid")
        np.testing.assert_allclose(contributions.sum(axis=1) + np.ravel(model.intercept_)[0],
            model.decision_function(transformed.iloc[[6, 7]]), atol=1e-12)
    else:
        metadata = json.loads((destination / "explanation_metadata.json").read_text())
        contributions = pd.read_csv(destination / "heldout_shap_values.csv").drop(columns="caseid")
        assert np.abs(contributions.to_numpy()).max() > 0  # Do not pass on a constant-tree fixture.
        np.testing.assert_allclose(contributions.sum(axis=1) + metadata["shap_base_value"],
            model.predict(transformed.iloc[[6, 7]], output_margin=True), rtol=1e-6, atol=1e-6)
        assert "raw model log-odds" in metadata["shap_output_scale"]
    assert len(titles) >= 2
    assert all(title.startswith("Diagnostic: ") for title in titles)


def test_explanation_bar_uses_display_labels_without_invented_statistic_units(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import reporting.fold_explanations as explanations
    labels = []
    def save(fig, *args, **kwargs):
        labels.extend(label.get_text() for label in fig.axes[0].get_yticklabels())
        plt.close(fig)
        return []
    monkeypatch.setattr(explanations, "save_report_figure_bundle", save)
    explanations._bar(pd.DataFrame({"feature": ["age", "Primus_AWP_CO_f1ecac"], "importance": [.2, .3]}),
                      "importance", tmp_path / "importance", "Importance")
    assert "Age (years)" in labels
    assert any("AWP" in label for label in labels)
    assert all("hPa" not in label for label in labels)


def test_legacy_case_bootstrap_does_not_claim_known_patient_count():
    candidate = _pred()
    candidate.df = candidate.df.drop(columns="subjectid")
    summary, samples = metrics.summarize([candidate], n_bootstrap=5, n_jobs=1, parallel_backend="threads")
    assert pd.isna(summary.n_patients.iloc[0])
    assert pd.notna(summary.auprc_lower.iloc[0])


def test_diagnostic_compute_profile_cannot_be_promoted_by_changing_flags_and_fold_counts(tmp_path):
    from data_preparation.cohort_scopes import selected_case_hash
    reference, identity = _saved_protocol(_pred(family="EBM"), tmp_path,
                                          diagnostic=False, outer_folds=5, inner_folds=5)
    # Even if every generic status/fold field claims a full run, the bounded
    # fitting profile remains diagnostic and must stop promotion before export.
    reference.df["outer_fold_id"] = [0, 1, 2, 3, 4, 0]
    reference.df["repeat_id"] = 0
    metadata_path = reference.path.parent.parent / "artifacts" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    bounded = "diagnostic_ebm_200_rounds_2_outer_bags_no_inner_bags"
    metadata["compute_profile"] = bounded
    metadata["validation_fingerprint"]["compute_profile"] = bounded
    metadata_path.write_text(json.dumps(metadata))
    manifest = {"cohort_scope": "main_shared", "validation_scheme": "nested_cv", "diagnostic": False,
        "study_identity": identity,
        "configurations": [{"model_type": "ebm", "outcome": "any_aki", "branch": "non_windowed", "feature_set": "preop_only"}],
        "branch_rosters": {"non_windowed": {"selected_case_hash": selected_case_hash(reference.df), "dataset_sha256": "frozen_modeling_csv"}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(metrics.ValidationError, match="Diagnostic compute profile cannot be promoted"):
        metrics._validate_run_manifest([reference], path)
    reference.df["diagnostic"] = True
    metadata["diagnostic"] = True
    metadata["validation_fingerprint"]["diagnostic"] = True
    metadata_path.write_text(json.dumps(metadata))
    manifest["diagnostic"] = True
    path.write_text(json.dumps(manifest))
    loaded = metrics._validate_run_manifest([reference], path)
    assert loaded[0].analysis_status == "diagnostic"
