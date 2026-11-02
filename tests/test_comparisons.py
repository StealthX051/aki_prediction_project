"""Full saved-prediction comparisons, with no model fitting or research inputs."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from results_recreation import metrics_summary as metrics
from results_recreation import comparisons
from reporting.decision_curve import net_benefit


def prediction_sets(tmp_path, *, outcomes=("any_aki",), family_count=3):
    sets = []
    labels = {"logistic_regression": "Logistic regression", "xgboost": "XGBoost", "ebm": "EBM"}
    truth = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    for outcome in outcomes:
        for i, family in enumerate(comparisons.FAMILIES[:family_count]):
            for j, feature in enumerate(comparisons.CORE):
                probabilities = np.clip(np.array([.2, .45, .4, .65, .6, .5, .3, .7])
                                        + (truth - .5) * (i + j) * .07, .01, .99)
                frame = pd.DataFrame({"caseid": np.arange(8), "subjectid": [10, 10, 20, 30, 40, 40, 50, 60],
                    "y_true": truth, "y_prob_calibrated": probabilities, "y_prob_raw": probabilities,
                    "y_pred_label": (probabilities >= .5).astype(int), "threshold": .5,
                    "outer_fold_id": [0, 0, 1, 1, 0, 0, 1, 1], "repeat_id": 0})
                sets.append(metrics.PredictionSet(tmp_path / outcome / family / feature / "predictions/test.csv",
                    frame, .5, outcome, "non_windowed", feature, labels[family], "nested",
                    "main_shared", "nested_cv", "diagnostic"))
    return sets


def calculate(sets, *, budget=24, backend="threads", workers=1, manifest=None):
    summary, bootstrap = metrics.summarize(sets, n_bootstrap=budget, bootstrap_seed=17,
        stratified_bootstrap=False, n_jobs=workers, parallel_backend=backend,
        delta_mode="reference", strict=True)
    result = comparisons.build_comparison_artifacts(sets, summary, bootstrap, n_bootstrap=budget,
        bootstrap_seed=17, stratified_bootstrap=False, run_manifest=manifest)
    return summary, bootstrap, result


def test_complete_core_has_unique_prespecified_contrasts_and_six_primary_ap_results(tmp_path):
    sets = prediction_sets(tmp_path, outcomes=("any_aki", "icu_admission"))
    # Inventory expansion does not need hundreds of bootstrap evaluations.
    _, _, result = calculate(sets, budget=0)
    inventory = result["inventory"]
    assert inventory["comparison_status"] == "complete"
    assert len(inventory["comparisons"]) == 36
    assert len({row["comparison_id"] for row in inventory["comparisons"]}) == 36
    paired = result["paired"]
    assert len(paired) == 36 * len(comparisons.METRICS)
    primary = paired.loc[paired.designation.eq("primary")]
    assert len(primary) == 6 and primary.metric.eq("auprc").all()
    assert primary.reference_feature_set.eq("preop_only").all()
    assert primary.candidate_feature_set.eq("preop_and_all_waveforms").all()
    assert result["delong"].status.eq("unsupported_design").all()
    assert result["delong"].p_value.isna().all()
    assert not result["delong"].comparison_id.duplicated().any()


def test_metric_and_dca_differences_use_the_same_operation_weighted_patient_draws(tmp_path):
    sets = prediction_sets(tmp_path, family_count=1)
    summary, _, result = calculate(sets, budget=40)
    assert set(result["dca"].bootstrap_draw_hash) == set(summary.bootstrap_draw_hash)
    ref, cand = sets[0], sets[2]
    rows = result["paired"].loc[lambda x: x.reference_feature_set.eq(ref.feature_set)
                                & x.candidate_feature_set.eq(cand.feature_set) & x.metric.eq("brier")]
    draws = metrics._generate_bootstrap_indices(ref.df, 40, False, 17)
    brier_delta, nb_delta = [], []
    for draw in draws:
        truth = ref.df.y_true.to_numpy()[draw]
        left, right = ref.df.y_prob_calibrated.to_numpy()[draw], cand.df.y_prob_calibrated.to_numpy()[draw]
        brier_delta.append(np.mean((right - truth) ** 2) - np.mean((left - truth) ** 2))
        nb_delta.append(net_benefit(truth, right, [.1]).model.iloc[0]
                        - net_benefit(truth, left, [.1]).model.iloc[0])
    np.testing.assert_allclose(rows[["lower", "upper"]].iloc[0], np.percentile(brier_delta, [2.5, 97.5]))
    nb = result["dca_comparisons"].loc[lambda x: x.reference_feature_set.eq(ref.feature_set)
               & x.candidate_feature_set.eq(cand.feature_set) & np.isclose(x.threshold_probability, .1)]
    np.testing.assert_allclose(nb[["lower", "upper"]].iloc[0], np.percentile(nb_delta, [2.5, 97.5]))
    assert nb.bootstrap_valid.iloc[0] == 40
    assert result["inventory"]["comparison_status"] == "partial"
    assert sum(x["availability"] == "available" for x in result["inventory"]["comparisons"]) == 3


def test_family_stage_ablation_inventory_and_manifest_completeness(tmp_path):
    sets = prediction_sets(tmp_path, family_count=1)
    sets.append(replace(sets[1], feature_set="pleth_only", path=tmp_path / "ablation/predictions/test.csv"))
    _, _, result = calculate(sets, budget=0)
    ablations = [x for x in result["inventory"]["comparisons"] if x["comparison_type"] == "ablation"]
    assert len(ablations) == 1 and ablations[0]["reference_feature_set"] == "preop_only"
    assert ablations[0]["designation"] == "exploratory"
    manifest = {"cohort_scope": "main_shared", "validation_scheme": "nested_cv", "configurations": [
        {"model_type": "ebm", "outcome": "any_aki", "branch": "non_windowed", "feature_set": "all_waveforms"}]}
    with pytest.raises(metrics.ValidationError, match="manifest-promised"):
        calculate(sets, budget=0, manifest=manifest)


@pytest.mark.parametrize("column,value", [("caseid", 99), ("subjectid", 99), ("y_true", 1),
                                         ("outer_fold_id", 9), ("repeat_id", 9)])
def test_comparison_rejects_mismatched_pairing(tmp_path, column, value):
    sets = prediction_sets(tmp_path, family_count=1)
    summary, bootstrap, _ = calculate(sets, budget=0)
    sets[1].df.loc[0, column] = value
    with pytest.raises(metrics.ValidationError, match="rosters differ"):
        comparisons.build_comparison_artifacts(sets, summary, bootstrap, n_bootstrap=0, bootstrap_seed=17)


def test_row_order_workers_and_backends_produce_identical_statistics(tmp_path):
    sets = prediction_sets(tmp_path, family_count=1)
    _, _, original = calculate(sets, budget=8)
    reordered = [replace(ps, df=ps.df.iloc[::-1].copy()) for ps in sets]
    for data, backend in ((reordered, "threads"), (sets, "processes")):
        _, _, changed = calculate(data, budget=8, workers=2, backend=backend)
        for key in comparisons.ARTIFACTS:
            pd.testing.assert_frame_equal(original[key], changed[key])


def test_comparisons_reject_altered_bootstrap_and_summary_provenance(tmp_path):
    sets = prediction_sets(tmp_path, family_count=1)
    summary, bootstrap, _ = calculate(sets)
    bootstrap.loc[0, "bootstrap_draw_hash"] = "changed"
    with pytest.raises(metrics.ValidationError, match="draws differ"):
        comparisons.build_comparison_artifacts(sets, summary, bootstrap, n_bootstrap=24, bootstrap_seed=17)
    summary.loc[0, "bootstrap_seed"] = 99
    with pytest.raises(metrics.ValidationError, match="provenance differ"):
        comparisons.build_comparison_artifacts(sets, summary, bootstrap, n_bootstrap=24, bootstrap_seed=17)


@pytest.mark.parametrize("artifact", [*comparisons.ARTIFACTS.values(), "metrics_summary.csv"])
def test_saved_comparisons_reject_changes_before_report_exports(tmp_path, artifact):
    sets = prediction_sets(tmp_path, family_count=1)
    summary, _, result = calculate(sets, budget=0)
    summary.to_csv(tmp_path / "metrics_summary.csv", index=False)
    comparisons.write_comparison_artifacts(result, tmp_path)
    loaded = comparisons.load_comparison_artifacts(tmp_path, sets, summary)
    assert loaded["inventory"]["comparison_status"] == "partial"
    assert len(loaded["paired"]) == 24
    path = tmp_path / artifact
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(metrics.ValidationError, match="altered|Stale"):
        comparisons.load_comparison_artifacts(tmp_path, sets, summary)


def test_saved_comparisons_reject_changed_predictions_policy_and_selection(tmp_path):
    sets = prediction_sets(tmp_path, family_count=1)
    summary, _, result = calculate(sets, budget=0)
    summary.to_csv(tmp_path / "metrics_summary.csv", index=False)
    comparisons.write_comparison_artifacts(result, tmp_path)
    assert result["paired"].lower.isna().all()
    assert result["paired"].interval_status.eq("not_estimated").all()
    assert result["dca"].model_lower.isna().all()
    with pytest.raises(metrics.ValidationError, match="resampling policies"):
        comparisons.load_comparison_artifacts(tmp_path, sets, summary.assign(bootstrap_seed=99))
    with pytest.raises(metrics.ValidationError, match="run selection"):
        comparisons.load_comparison_artifacts(tmp_path, sets, summary, run_manifest={"configurations": []})
    sets[0].df.loc[0, "y_prob_calibrated"] = .33
    with pytest.raises(metrics.ValidationError, match="prediction provenance"):
        comparisons.load_comparison_artifacts(tmp_path, sets, summary)


def test_manifest_reporting_policy_is_enforced_and_changes_invalidate_statistics(tmp_path):
    sets = prediction_sets(tmp_path, family_count=1)
    manifest = {"cohort_scope": "main_shared", "validation_scheme": "nested_cv", "configurations": [],
                "reporting": {"bootstrap_replicates": 8, "bootstrap_seed": 17, "bootstrap_stratified": False}}
    summary, bootstrap, result = calculate(sets, budget=8, manifest=manifest)
    summary.to_csv(tmp_path / "metrics_summary.csv", index=False)
    comparisons.write_comparison_artifacts(result, tmp_path)
    comparisons.load_comparison_artifacts(tmp_path, sets, summary, run_manifest=manifest)
    changed = json.loads(json.dumps(manifest))
    changed["reporting"]["bootstrap_replicates"] = 5000
    with pytest.raises(metrics.ValidationError, match="run selection"):
        comparisons.load_comparison_artifacts(tmp_path, sets, summary, run_manifest=changed)
    with pytest.raises(metrics.ValidationError, match="differs from run manifest"):
        comparisons.build_comparison_artifacts(sets, summary, bootstrap, n_bootstrap=8,
                                              bootstrap_seed=17, run_manifest=changed)


def test_comparison_engine_never_calls_missing_patients_known_patients_in_legacy_mode(tmp_path):
    sets = prediction_sets(tmp_path, family_count=1)
    summary, bootstrap, _ = calculate(sets, budget=0)
    sets = [replace(ps, df=ps.df.drop(columns="subjectid")) for ps in sets]
    with pytest.raises(metrics.ValidationError, match="Patient IDs"):
        comparisons.build_comparison_artifacts(sets, summary, bootstrap, n_bootstrap=0,
                                              bootstrap_seed=17, strict=False)


def test_mortality_comparisons_are_exploratory_and_use_rare_outcome_grid(tmp_path):
    sets = [replace(ps, cohort_scope="mortality_waveform_sensitivity")
            for ps in prediction_sets(tmp_path, outcomes=("mortality",))]
    _, _, result = calculate(sets, budget=0)
    assert set(result["paired"].designation) == {"exploratory"}
    assert result["dca"].threshold_probability.nunique() == 50
    assert result["dca"].threshold_probability.min() == .001
    assert result["dca"].threshold_probability.max() == .05
    assert result["inventory"]["dca_ranges"]["mortality"] == (np.arange(1, 51) / 1000).tolist()
    assert result["delong"].status.eq("unsupported_design").all()
