"""Guarded DeLong inference and independent pinned Sun/Xu fixture values."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from data_preparation.cohort_scopes import selected_case_hash
from results_recreation.delong import add_holm_adjustment, gated_delong, paired_delong


# Values independently generated with yandexdataschool/roc_comparison's
# compare_auc_delong_xu.py, using its midrank covariance implementation.
# Pinned commit 62da5b8f59fb9029dca8365d2b7098de735dc658, source blob
# c615e124c97b5103a03ed3d39406105a8ebbe0a5:
# https://github.com/yandexdataschool/roc_comparison/blob/62da5b8f59fb9029dca8365d2b7098de735dc658/compare_auc_delong_xu.py
@pytest.mark.parametrize("truth,reference,candidate,expected", [
    ([0, 0, 0, 0, 1, 1, 1, 1], [.1, .4, .35, .8, .5, .7, .6, .9],
     [.2, .3, .6, .7, .8, .5, .4, .9], [.8125, .75, .0390625, .7518296340458492]),
    ([0, 1, 0, 1, 0, 1, 0, 1], [.1, .4, .4, .6, .8, .6, .4, .9],
     [.2, .5, .4, .5, .7, .8, .4, .8], [.75, .875, .0078125, .15729920705028533]),
    ([0, 0, 1, 1, 0, 1], [.1, .2, .2, .3, .3, .4],
     [.4, .1, .3, .2, .2, .5], [7 / 9, 13 / 18, .11728395061728394, .8711314915971587]),
])
def test_delong_agrees_with_independent_midrank_reference(truth, reference, candidate, expected):
    result = paired_delong(truth, reference, candidate)
    np.testing.assert_allclose([result[key] for key in ("reference_auc", "candidate_auc", "variance", "p_value")],
                               expected, rtol=0, atol=1e-12)
    assert result["status"] == "ok"
    assert result["delta"] == pytest.approx(expected[1] - expected[0])
    assert np.exp(result["log_p_value"]) == pytest.approx(result["p_value"])


@pytest.mark.parametrize("truth,reference,candidate", [
    ([], [], []), ([0, 0, 1], [.1, .2], [.1, .2, .3]),
    ([[0], [1]], [.1, .2], [.2, .1]), ([0, 0, 1, 2], [.1] * 4, [.2] * 4),
    ([0, 0, 1, .5], [.1] * 4, [.2] * 4), ([0, 0, 1, 1], [.1, .2, np.nan, .9], [.2] * 4),
    ([0, 0, 1, 1], [.1] * 4, [.2, np.inf, .5, .6]),
    ([0, 0, 1], [.1, .4, .9], [.1, .9, .4]), ([0, 0, 0], [.1] * 3, [.2] * 3),
])
def test_delong_rejects_invalid_vectors_without_hanging(truth, reference, candidate):
    with pytest.raises(ValueError):
        paired_delong(truth, reference, candidate)


@pytest.mark.parametrize("candidate", [[.1, .2, .8, .9], [.9, .8, .2, .1], [.5] * 4])
def test_zero_variance_is_explicit_and_never_reassuring_p_one(candidate):
    result = paired_delong([0, 0, 1, 1], [.1, .2, .8, .9], candidate)
    assert result["status"] == "degenerate"
    assert result["variance"] == 0
    assert np.isnan(result["p_value"]) and np.isnan(result["log_p_value"])


def test_tail_probability_and_direction_are_stable_without_mutating_inputs():
    truth = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]
    reference = np.random.default_rng(88).normal(size=len(truth))
    candidate = truth.astype(float)
    originals = (truth.copy(), reference.copy(), candidate.copy())
    result = paired_delong(truth, reference, candidate)
    reverse = paired_delong(truth, candidate, reference)
    assert result["p_value"] == pytest.approx(3.280292440064008e-47, rel=1e-12, abs=0)
    assert result["delta"] == -reverse["delta"]
    assert result["p_value"] == reverse["p_value"]
    assert result["candidate_auc"] == roc_auc_score(truth, candidate)
    for current, original in zip((truth, reference, candidate), originals):
        np.testing.assert_array_equal(current, original)


def _prediction(tmp_path, name, *, scheme="holdout"):
    frame = pd.DataFrame({
        "caseid": range(1, 9), "subjectid": range(101, 109),
        "y_true": [0, 0, 0, 0, 1, 1, 1, 1],
        "y_prob_calibrated": [.1, .4, .35, .8, .5, .7, .6, .9],
        "outer_fold_id": 0, "repeat_id": 0, "fold": 0,
    })
    path = tmp_path / name / "predictions" / "test.csv"
    return SimpleNamespace(path=path, df=frame, cohort_scope="main_shared", outcome="any_aki",
                           branch="non_windowed", validation_scheme=scheme)


def _save_holdout(ps):
    ps.path.parent.mkdir(parents=True, exist_ok=True)
    artifact = ps.path.parent.parent / "artifacts"
    artifact.mkdir()
    training = pd.DataFrame({"caseid": [20, 21, 22, 23], "subjectid": [120, 121, 122, 123], "y_true": [0, 0, 1, 1]})
    training.to_csv(ps.path.parent / "train_oof.csv", index=False)
    ps.df.to_csv(ps.path, index=False)
    fingerprint = {"cohort_scope": "main_shared", "validation_scheme": "holdout",
                   "selected_case_hash": selected_case_hash(pd.concat([training, ps.df], ignore_index=True)),
                   "fold_assignment_hash": "fixed-patient-holdout", "dataset_hash": "same-source"}
    (artifact / "metadata.json").write_text(json.dumps({"validation_scheme": "holdout", "validation_fingerprint": fingerprint}))


def test_gate_accepts_only_saved_independent_holdout_and_reorders_cases(tmp_path):
    reference, candidate = _prediction(tmp_path, "reference"), _prediction(tmp_path, "candidate")
    candidate.df["y_prob_calibrated"] = [.2, .3, .6, .7, .8, .5, .4, .9]
    _save_holdout(reference)
    _save_holdout(candidate)
    candidate.df = candidate.df.iloc[::-1].reset_index(drop=True)
    result = gated_delong(reference, candidate)
    assert result["status"] == "ok"
    assert result["p_value"] == pytest.approx(.7518296340458492)
    assert result["n_patients"] == result["n_operations"] == 8


@pytest.mark.parametrize("change,reason", [
    ("nested_cv", "pooled_cv"), ("repeat_patient", "repeated_patient"),
    ("no_provenance", "missing_holdout_provenance"), ("multiple_folds", "multiple_holdout"),
    ("training_overlap", "training_test_overlap"), ("changed_saved_scores", "saved_test_predictions"),
    ("changed_fold_hash", "provenance_mismatch"), ("changed_roster_hash", "selected_case_hash_mismatch"),
])
def test_gate_refuses_unsupported_inference(tmp_path, change, reason):
    reference, candidate = _prediction(tmp_path, "reference"), _prediction(tmp_path, "candidate")
    _save_holdout(reference)
    _save_holdout(candidate)
    if change == "nested_cv":
        reference.validation_scheme = candidate.validation_scheme = "nested_cv"
    elif change == "repeat_patient":
        reference.df.loc[1, "subjectid"] = candidate.df.loc[1, "subjectid"] = 101
    elif change == "no_provenance":
        candidate.path.unlink()
    elif change == "multiple_folds":
        reference.df.loc[0, "outer_fold_id"] = candidate.df.loc[0, "outer_fold_id"] = 1
    elif change == "training_overlap":
        path = candidate.path.parent / "train_oof.csv"
        training = pd.read_csv(path)
        training.loc[0, "subjectid"] = 101
        training.to_csv(path, index=False)
    elif change == "changed_saved_scores":
        candidate.df.loc[0, "y_prob_calibrated"] = .9
    else:
        path = candidate.path.parent.parent / "artifacts" / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["validation_fingerprint"]["fold_assignment_hash" if change == "changed_fold_hash" else "selected_case_hash"] = "changed"
        path.write_text(json.dumps(metadata))
    result = gated_delong(reference, candidate)
    assert result["status"] == "unsupported_design"
    assert reason in result["reason"]
    assert np.isnan(result["p_value"]) and np.isnan(result["log_p_value"])


@pytest.mark.parametrize("column", ["caseid", "subjectid", "y_true", "outer_fold_id", "repeat_id"])
def test_gate_rejects_mispaired_identity_even_for_cv(tmp_path, column):
    reference, candidate = _prediction(tmp_path, "reference", scheme="nested_cv"), _prediction(tmp_path, "candidate", scheme="nested_cv")
    candidate.df.loc[0, column] += 1 if column != "caseid" else 20
    with pytest.raises(ValueError, match="identities"):
        gated_delong(reference, candidate)


def test_holm_adjusts_unique_eligible_pairs_per_population_in_log_space():
    frame = pd.DataFrame({"comparison_id": ["a", "b", "c", "d", "e"], "status": ["ok"] * 4 + ["unsupported_design"],
                          "outcome": ["aki"] * 3 + ["icu", "aki"], "p_value": [.01, .04, .03, .02, np.nan],
                          "log_p_value": np.log([.01, .04, .03, .02, np.nan])})
    adjusted = add_holm_adjustment(frame)
    np.testing.assert_allclose(adjusted.p_value_holm[:4], [.03, .06, .06, .02])
    assert np.isnan(adjusted.p_value_holm.iloc[4])
    assert "p_value_holm" not in frame
    tiny = pd.DataFrame({"status": ["ok", "ok"], "p_value": [0., .2], "log_p_value": [-1000., np.log(.2)]})
    result = add_holm_adjustment(tiny)
    assert result.log_p_value_holm.iloc[0] == pytest.approx(-1000 + np.log(2))
    raw = add_holm_adjustment(frame.drop(columns="log_p_value"))
    np.testing.assert_allclose(raw.p_value_holm, adjusted.p_value_holm, equal_nan=True)
    with pytest.raises(ValueError, match="unique comparison"):
        add_holm_adjustment(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
