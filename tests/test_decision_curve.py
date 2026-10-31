"""Binary DCA and operation-weighted patient-bootstrap reference checks."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from reporting import decision_curve as dca


def test_binary_dca_reference_fixture_with_exact_threshold_ties():
    # Pinned oracle: MSKCC-Epi-Bio/dcurves 1.1.7, commit
    # 168c8b654808691d70a1ea6b92d2a5013915efc2. Higher risk includes ties (>=).
    table = dca.net_benefit([0, 1, 1, 0, 0, 1], [0, .25, .5, .5, 1, .75], [.25, .5, .75])
    np.testing.assert_allclose(table.model, [7 / 18, 0, -1 / 3], atol=1e-12)
    np.testing.assert_allclose(table.treat_all, [1 / 3, 0, -1], atol=1e-12)
    np.testing.assert_array_equal(table.treat_none, 0)


@pytest.mark.parametrize("probability,expected", [(0, [0, 0, 0]), (.5, [1 / 3, 0, 0]), (1, [1 / 3, 0, -1])])
def test_constant_prediction_reference_values(probability, expected):
    table = dca.net_benefit([0, 1], [probability] * 2, [.25, .5, .75])
    np.testing.assert_allclose(table.model, expected, atol=1e-12)


def test_decision_thresholds_must_be_finite_and_strictly_between_zero_and_one():
    for invalid in ([0], [1], [np.nan]):
        with pytest.raises(ValueError, match="strictly between"):
            dca.net_benefit([0, 1], [.2, .8], invalid)


def test_patient_draw_counts_keep_repeat_operations_and_operation_denominators():
    groups = [20, 20, 10, 30, 30]
    truth, risk, cutoffs = np.array([1, 0, 0, 1, 1]), np.array([.8, .2, .7, .6, .1]), [.25, .5, .75]
    indices = [np.array([0, 1, 0, 1, 2]), np.array([2, 2, 2]),
               np.array([3, 4, 3, 4, 3, 4]), np.arange(5)]
    counts = dca.cluster_resample_counts(groups, indices)
    np.testing.assert_array_equal(counts, [[2, 1, 0], [0, 3, 0], [0, 0, 3], [1, 1, 1]])
    assert counts.dtype == np.int32
    table, draws = dca.bootstrap_net_benefit(truth, risk, cutoffs, groups=groups, cluster_counts=counts)
    expected = np.array([dca.net_benefit(truth[index], risk[index], cutoffs).model for index in indices])
    expected_all = np.array([dca.net_benefit(truth[index], risk[index], cutoffs).treat_all for index in indices])
    np.testing.assert_allclose(draws, expected, atol=1e-12)
    np.testing.assert_allclose(table.model_lower, np.percentile(expected, 2.5, axis=0), atol=1e-12)
    np.testing.assert_allclose(table.treat_all_upper, np.percentile(expected_all, 97.5, axis=0), atol=1e-12)
    assert table.model_bootstrap_valid.eq(4).all()
    # Entirely negative and entirely positive draws remain defined.
    assert np.isfinite(draws).all()
    assert draws[1, 1] == -1
    assert draws[2, 1] == .5


def test_batched_bootstrap_agrees_with_raw_draws_and_shared_paired_differences():
    groups, truth = np.repeat([9, 4, 7], [2, 1, 2]), np.array([0, 1, 0, 1, 1])
    left, right = np.array([.1, .6, .7, .8, .3]), np.array([.2, .9, .4, .8, .6])
    by_patient = [np.flatnonzero(groups == patient) for patient in [9, 4, 7]]
    rng = np.random.default_rng(27)
    indices = [np.concatenate([by_patient[patient] for patient in rng.integers(0, 3, 3)]) for _ in range(129)]
    counts = dca.cluster_resample_counts(groups, indices)
    _, left_draws = dca.bootstrap_net_benefit(truth, left, [.1, .5, .9], groups=groups, cluster_counts=counts)
    _, right_draws = dca.bootstrap_net_benefit(truth, right, [.1, .5, .9], groups=groups, cluster_counts=counts)
    differences = [dca.net_benefit(truth[index], right[index], [.1, .5, .9]).model.to_numpy()
                   - dca.net_benefit(truth[index], left[index], [.1, .5, .9]).model.to_numpy() for index in indices]
    np.testing.assert_allclose(right_draws - left_draws, differences, atol=1e-12)


@pytest.mark.parametrize("indices", [[np.array([0, 2])], [np.array([0, 1])], [np.array([0., 1., 2.])],
                                    [np.array([-1, 0, 1])], [np.array([0, 1, 3])], [np.array([], dtype=int)]])
def test_invalid_or_split_patient_draws_are_rejected(indices):
    with pytest.raises(ValueError):
        dca.cluster_resample_counts([5, 5, 8], indices)


def test_empty_bootstrap_has_no_interval_and_patient_identity_is_required():
    counts = dca.cluster_resample_counts([7, 8], [])
    assert counts.shape == (0, 2)
    table, draws = dca.bootstrap_net_benefit([0, 1], [.1, .8], [.25], groups=[7, 8], cluster_counts=counts)
    assert draws.shape == (0, 1)
    assert table.model_lower.isna().all() and table.model_upper.isna().all()
    assert table.model_bootstrap_valid.eq(0).all()
    with pytest.raises(ValueError, match="patient identities"):
        dca.cluster_resample_counts([7, None], [])
    with pytest.raises(ValueError, match="whole-patient"):
        dca.bootstrap_net_benefit([0, 1], [.1, .8], [.25], groups=[7, 8], cluster_counts=[[.5, 1.5]])


def test_outcome_grids_are_prespecified_and_include_decimal_boundaries():
    np.testing.assert_allclose(dca.outcome_thresholds("any_aki"), np.arange(1, 31) / 100)
    np.testing.assert_allclose(dca.outcome_thresholds("icu_admission"), np.arange(5, 61) / 100)
    with pytest.raises(ValueError, match="No exploratory"):
        dca.outcome_thresholds("unknown")


def _saved_curves():
    summary, pairs = [], []
    for family, label in [("logistic_regression", "Logistic regression"), ("xgboost", "XGBoost")]:
        for feature in dca.CORE_FEATURE_SET_ORDER:
            table, _ = dca.bootstrap_net_benefit([0, 1], [.1, .8], [.1, .2], groups=[1, 2],
                                              cluster_counts=np.array([[2, 0], [1, 1], [0, 2]], dtype=np.int32))
            table = table.assign(cohort_scope="main_shared", validation_scheme="nested_cv", outcome="any_aki",
                branch="non_windowed", model_type=family, model_name=label, feature_set=feature,
                analysis_status="diagnostic", n_operations=2, n_patients=2, bootstrap_replicates=3,
                bootstrap_seed=0, bootstrap_stratified=False, prediction_content_hash=family + feature)
            summary.append(table)
        for threshold in [.1, .2]:
            pairs.append(dict(comparison_id=family, reference_model=family, candidate_model=family,
                reference_feature_set="preop_only", candidate_feature_set="preop_and_all_waveforms",
                threshold_probability=threshold, delta=.02, lower=-.01, upper=.04, bootstrap_valid=3,
                analysis_status="diagnostic", cohort_scope="main_shared", validation_scheme="nested_cv",
                outcome="any_aki", branch="non_windowed"))
    return pd.concat(summary, ignore_index=True), pd.DataFrame(pairs)


def test_saved_renderer_keeps_basenames_status_order_and_pointwise_delta_bands(tmp_path, monkeypatch):
    summary, pairs = _saved_curves()
    figures = {}
    def save(fig, path, **kwargs):
        figures[path.name] = [(ax.get_title(), len(ax.collections), [line.get_label() for line in ax.lines]) for ax in fig.axes]
        plt.close(fig)
        return []
    monkeypatch.setattr(dca, "save_report_figure_bundle", save)
    paths = dca.render_comparison_curves(summary, pairs, tmp_path, comparison_status="partial")
    assert "dca_any_aki_non_windowed_XGBoost" in figures
    assert "dca_delta_any_aki_non_windowed_xgboost" in figures
    assert "dca_families_any_aki_non_windowed_preop_only" in figures
    assert figures["dca_delta_any_aki_non_windowed_xgboost"][0][1] == 1
    primary = figures["dca_any_aki_non_windowed_XGBoost"][0]
    assert "[diagnostic]" in primary[0] and "partial comparisons" in primary[0]
    assert primary[2][-2:] == ["Treat all", "Treat none"]
    metadata = json.loads((tmp_path / "dca_any_aki_non_windowed_XGBoost.json").read_text())
    assert metadata["bootstrap_replicates"] == 3 and metadata["n_operations"] == 2
    assert metadata["comparison_status"] == "partial"
    assert "pointwise" in metadata["uncertainty"]
    assert metadata["prediction_provenance"]
    assert paths and all(path.exists() for path in paths)


def test_saved_renderer_rejects_duplicate_or_mixed_curves_before_writing(tmp_path):
    summary, pairs = _saved_curves()
    root = tmp_path / "new"
    with pytest.raises(ValueError, match="Duplicate"):
        dca.render_comparison_curves(pd.concat([summary, summary.iloc[:1]]), pairs, root)
    assert not root.exists()
    summary.loc[0, "cohort_scope"] = "icu_waveform_sensitivity"
    with pytest.raises(ValueError, match="mix cohort_scope"):
        dca.render_comparison_curves(summary, pairs, root)
    assert not root.exists()


def test_actual_saved_png_svg_exports_do_not_recalculate_statistics(tmp_path, monkeypatch):
    summary, pairs = _saved_curves()
    summary = summary.loc[summary.model_type.eq("xgboost")]
    pairs = pairs.loc[pairs.candidate_model.eq("xgboost")]
    monkeypatch.setattr(dca, "net_benefit", lambda *args, **kwargs: pytest.fail("Rendering recalculated DCA"))
    paths = dca.render_comparison_curves(summary, pairs, tmp_path, comparison_status="partial")
    assert all(path.exists() for path in paths)
    import matplotlib.image as mpimg
    png = tmp_path / "dca_delta_any_aki_non_windowed_xgboost.png"
    height, width = mpimg.imread(png).shape[:2]
    assert height > 300 and width / height < 3.2  # Long labels must not expand the canvas.
    svg = png.with_suffix(".svg").read_text()
    assert "95% pointwise intervals" in svg and "partial comparisons" in svg


def test_full_grid_keeps_main_curves_core_only_but_preserves_ablation_numbers(tmp_path, monkeypatch):
    summary, pairs = _saved_curves()
    ablations = summary.loc[summary.feature_set.eq("preop_only")].assign(feature_set="ecg_only")
    summary = pd.concat([ablations, summary.iloc[::-1]], ignore_index=True)
    figures = {}
    def save(fig, path, **kwargs):
        figures[path.name] = [line.get_label() for line in fig.axes[0].lines]
        plt.close(fig)
        return []
    monkeypatch.setattr(dca, "save_report_figure_bundle", save)
    dca.render_comparison_curves(summary, pairs, tmp_path)
    stem = "dca_any_aki_non_windowed_XGBoost"
    labels = dca.load_display_dictionary().feature_set_labels()
    assert figures[stem] == [labels[feature] for feature in dca.CORE_FEATURE_SET_ORDER] + ["Treat all", "Treat none"]
    assert "ecg_only" in set(pd.read_csv(tmp_path / f"{stem}.csv").feature_set)
    assert json.loads((tmp_path / f"{stem}.json").read_text())["plotted_feature_sets"] == list(dca.CORE_FEATURE_SET_ORDER)


def test_delta_panels_follow_stable_input_and_family_order(tmp_path, monkeypatch):
    summary, pairs = _saved_curves()
    extra = []
    for family in ("xgboost", "logistic_regression"):
        sample = pairs.loc[pairs.candidate_model.eq(family)]
        extra.extend([
            sample.assign(comparison_id=family + "wave", candidate_feature_set="all_waveforms"),
            sample.assign(comparison_id=family + "clinical", reference_feature_set="all_waveforms"),
        ])
    pairs = pd.concat([*extra, pairs.iloc[::-1]], ignore_index=True)
    figures = {}
    def save(fig, path, **kwargs):
        figures[path.name] = [ax.get_title() for ax in fig.axes]
        plt.close(fig)
        return []
    monkeypatch.setattr(dca, "save_report_figure_bundle", save)
    dca.render_comparison_curves(summary, pairs, tmp_path)
    labels = dca.load_display_dictionary().feature_set_labels()
    first = figures["dca_delta_any_aki_non_windowed_xgboost"]
    assert labels["all_waveforms"] in first[0] and labels["preop_only"] in first[0]
    assert labels["preop_and_all_waveforms"] in first[1] and labels["preop_only"] in first[1]
    assert labels["preop_and_all_waveforms"] in first[2] and labels["all_waveforms"] in first[2]
    names = list(figures)
    assert names.index("dca_any_aki_non_windowed_Logistic regression") < names.index("dca_any_aki_non_windowed_XGBoost")
