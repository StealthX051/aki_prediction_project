"""Finite-fallback audit uses only synthetic arrays and the pinned extractor."""
import numpy as np
import pycatch22

from scripts.window_duration_failures import (
    EMBEDDING, FLUCTUATIONS, embedding_fallback_status, fluctuation_scale_count,
    verified_fallbacks,
)


def test_regular_increment_series_verifies_embedding_zero_guard():
    signal = np.arange(1000, dtype=float)
    actual = pycatch22.catch22_all(signal, catch24=True)
    index = actual["names"].index(EMBEDDING)
    assert actual["values"][index] == 0
    detail = embedding_fallback_status(signal)
    assert detail["verified"] and detail["distance_sd"] < 0.001
    mask = verified_fallbacks(signal, actual["values"], actual["names"])
    assert mask[index]
    assert not mask[actual["names"].index("PD_PeriodicityWang_th0_01")]


def test_zero_alone_is_not_a_verified_embedding_failure():
    signal = np.random.default_rng(17).standard_normal(1000)
    actual = pycatch22.catch22_all(signal, catch24=True)
    index = actual["names"].index(EMBEDDING)
    assert not embedding_fallback_status(signal)["verified"]
    values = np.array(actual["values"])
    values[index] = 0  # Deliberately ambiguous output; raw-zero inference is forbidden.
    assert not verified_fallbacks(signal, values, actual["names"])[index]


def test_fluctuation_fallthrough_is_outside_successful_split_range():
    assert fluctuation_scale_count(625) == 47
    assert fluctuation_scale_count(1000) == 48
    assert fluctuation_scale_count(64000) == 50
    names = sorted(FLUCTUATIONS) + ["PD_PeriodicityWang_th0_01"]
    signal = np.random.default_rng(1).standard_normal(625)
    assert verified_fallbacks(signal, [1 / 47, 6 / 47, 0], names).tolist() == [True, False, False]
