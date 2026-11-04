"""Synthetic invariants for feature-only window-origin screening."""
import json

import numpy as np
import pytest

from scripts.window_duration_summary import score_aggregates, select_candidate, summarize


def _stable(n=120, channels=1, lengths=3):
    baseline = np.arange(n, dtype=float).reshape(n, 1, 1, 1, 1, 1)
    origins = np.arange(4, dtype=float).reshape(1, 1, 1, 4, 1, 1) * 0.01
    return np.broadcast_to(baseline + origins, (n, channels, lengths, 4, 4, 24)).copy()


def test_stable_scaled_outputs_and_constant_descriptor_fail_individually():
    values = _stable()
    values[..., 0] = 7  # One unevaluable descriptor must not fail the length.
    point = score_aggregates(values)
    assert not point["evaluable"][..., 0].any()
    assert (point["origin_pass_counts"] == 23).all()
    assert select_candidate([10, 20, 40], point["availability_pass_counts"], point["origin_pass_counts"]) == 10
    scaled = score_aggregates(values * -1000)
    np.testing.assert_allclose(point["p90_normalized_range"], scaled["p90_normalized_range"], equal_nan=True)
    np.testing.assert_array_equal(point["pass"], scaled["pass"])


def test_missing_origin_propagates_without_dropping_patient_denominator():
    values = _stable()
    values[:6, 0, :, 0, 0, 0] = np.nan
    point = score_aggregates(values)
    assert (point["n_available"][..., 0] == 114).all()
    assert point["availability_pass"][..., 0].all()
    values[6, 0, :, 0, 0, 0] = np.inf
    point = score_aggregates(values)
    assert not point["availability_pass"][..., 0].any()
    assert not point["evaluable"][0, :, 0, 0].any()
    assert point["evaluable"][0, :, 1, 0].all()


def test_origin_error_quantile_and_nonpassing_short_length():
    values = _stable()
    values[:, 0, 0, 3, :, :] += 100
    point = score_aggregates(values)
    assert (point["origin_pass_counts"][:, 0] == 0).all()
    expected = 100.03 / 59.5
    np.testing.assert_allclose(point["p90_normalized_range"][:, 0], expected)
    assert select_candidate([10, 20, 40], point["availability_pass_counts"], point["origin_pass_counts"]) == 20


def test_plateau_rejects_large_gain_and_untested_upper_neighbor():
    availability = np.full((1, 3), 24)
    origin = np.array([[[20] * 4, [23] * 4, [24] * 4]])
    assert select_candidate([160, 320, 640], availability, origin) == 320
    origin[:, :2] = 19
    assert select_candidate([160, 320, 640], availability, origin) is None
    origin[:] = 24
    availability[:] = [22, 24, 24]
    assert select_candidate([160, 320, 640], availability, origin) == 320


def test_bootstrap_is_reproducible_and_undefined_descriptor_does_not_fail_draw():
    values = _stable(n=24)
    values[..., 0] = np.nan
    summary = summarize(values, [10, 20, 40], bootstrap=4, seed=0)[0]
    assert summary["decision"] == "stable_technical_candidate"
    assert summary["bootstrap"]["selection_counts"]["10"] == 4
    assert summary["bootstrap"]["selection_counts"]["no_selection"] == 0
    json.dumps(summary, allow_nan=False)
    again = summarize(values, [10, 20, 40], bootstrap=4, seed=0)[0]
    assert summary == again


def test_empty_constant_and_all_missing_inputs_are_not_false_successes():
    with pytest.raises(ValueError, match="nonempty"):
        score_aggregates(np.empty((0, 1, 3, 4, 4, 24)))
    for value in [0, np.nan]:
        values = np.full((8, 1, 2, 4, 4, 24), value)
        summary = summarize(values, [10, 20], bootstrap=3)[0]
        assert summary["decision"] == "inconclusive"
        assert summary["technical_candidate_seconds"] is None
        assert summary["bootstrap"]["selection_counts"]["no_selection"] == 3
        json.dumps(summary, allow_nan=False)


def test_weighted_bootstrap_quantiles_match_explicit_patient_resampling():
    from scripts.window_duration_summary import _prepare, _score
    values = _stable(n=17)
    values[:4, 0, 0, 0, 0, 3] = np.nan
    values[:, :, :, 2] += np.arange(17).reshape(17, 1, 1, 1, 1) * 0.2
    indices = np.random.default_rng(12).integers(17, size=17)
    weighted = _score(_prepare(values), np.bincount(indices, minlength=17))
    explicit = score_aggregates(values[indices])
    for key in ["n_complete", "n_available", "scale", "p90_abs_range", "p90_normalized_range", "pass"]:
        np.testing.assert_allclose(weighted[key], explicit[key], equal_nan=True)
