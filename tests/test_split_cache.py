import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from model_creation.split_cache import cached_grouped_splits


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.delenv("AKI_SPLIT_CACHE_DIR", raising=False)
    y = np.tile([0, 1], 12)
    groups = np.repeat(np.arange(12), 2)
    return y, groups


def _compute(y, groups, folds=3, seed=42):
    splits = tuple(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(y[:, None], y, groups))
    _, encoded = np.unique(y, return_inverse=True)
    target = np.bincount(encoded) / folds
    score = float(sum(np.square((np.bincount(encoded[test], minlength=2) - target) / target).sum()
                      for _, test in splits))
    return splits, seed, score


def _call(y, groups, root, compute, **kwargs):
    return cached_grouped_splits(y, groups, kwargs.pop("folds", 3), kwargs.pop("seed", 42),
                                policy=kwargs.pop("policy", "test_v1"), compute=compute,
                                cache_dir=root, **kwargs)


def _assert_equal(left, right):
    assert left[1:] == right[1:]
    assert len(left[0]) == len(right[0])
    for actual, expected in zip(left[0], right[0]):
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def test_cache_hit_skips_splitter_and_returns_fresh_arrays(inputs, tmp_path, monkeypatch):
    y, groups = inputs
    expected = _call(y, groups, tmp_path, lambda: _compute(y, groups))
    monkeypatch.setattr(StratifiedGroupKFold, "split", lambda *a, **k: pytest.fail("cache hit called SGKF"))
    actual = _call(y, groups, tmp_path, lambda: pytest.fail("cache hit recomputed"))
    _assert_equal(actual, expected)
    actual[0][0][0][0] = -100
    _assert_equal(_call(y, groups, tmp_path, lambda: pytest.fail("recomputed")), expected)
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("tamper", ["checksum", "overlap", "patient_leakage", "score", "seed", "fractional_seed", "fractional_indices"])
def test_invalid_cache_recomputes_even_with_forged_checksum(inputs, tmp_path, tamper):
    y, groups = inputs
    expected = _call(y, groups, tmp_path, lambda: _compute(y, groups))
    path, = tmp_path.glob("*.json")
    envelope = json.loads(path.read_text())
    payload = envelope["payload"]
    if tamper == "checksum":
        envelope["sha256"] = "invalid"
    else:
        if tamper == "overlap":
            payload["test_indices"][1] = payload["test_indices"][0]
        elif tamper == "patient_leakage":
            payload["test_indices"][0][0], payload["test_indices"][1][0] = (
                payload["test_indices"][1][0], payload["test_indices"][0][0])
            payload["test_indices"][0].sort()
            payload["test_indices"][1].sort()
        elif tamper == "score":
            payload["score"] += 1
        elif tamper == "seed":
            payload["selected_seed"] = 200
        elif tamper == "fractional_seed":
            payload["selected_seed"] = 42.5
        else:
            payload["test_indices"][0][0] = 0.5
        envelope["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(envelope))
    calls = []
    actual = _call(y, groups, tmp_path, lambda: (calls.append(1), _compute(y, groups))[1])
    assert calls == [1]
    _assert_equal(actual, expected)
    _assert_equal(_call(y, groups, tmp_path, lambda: pytest.fail("repaired cache not usable")), expected)
    assert not list(tmp_path.glob("*.tmp"))


def test_input_order_and_policy_are_part_of_cache_identity(inputs, tmp_path):
    y, groups = inputs
    for order, policy in [(np.arange(len(y)), "v1"), (np.arange(len(y))[::-1], "v1"),
                          (np.arange(len(y)), "v2")]:
        ordered_y, ordered_groups = y[order], groups[order]
        _call(ordered_y, ordered_groups, tmp_path, lambda: _compute(ordered_y, ordered_groups), policy=policy)
    assert len(list(tmp_path.glob("*.json"))) == 3


@pytest.mark.parametrize("change", ["labels", "patients", "folds", "seed"])
def test_configuration_changes_cannot_reuse_prior_partitions(inputs, tmp_path, change):
    y, groups = inputs
    _call(y, groups, tmp_path, lambda: _compute(y, groups))
    folds, seed = 3, 42
    if change == "labels":
        y = 1 - y
    elif change == "patients":
        groups = groups + 100
    elif change == "folds":
        folds = 4
    else:
        seed = 43
    calls = []
    _call(y, groups, tmp_path,
          lambda: (calls.append(1), _compute(y, groups, folds, seed))[1], folds=folds, seed=seed)
    assert calls == [1]
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_interrupted_atomic_publish_leaves_no_partial_cache(inputs, tmp_path, monkeypatch):
    y, groups = inputs
    def fail_replace(*args):
        raise OSError("interrupted publish")
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted publish"):
        _call(y, groups, tmp_path, lambda: _compute(y, groups))
    assert not list(tmp_path.iterdir())


def test_cache_disabled_has_no_filesystem_effect(inputs, tmp_path, monkeypatch):
    y, groups = inputs
    monkeypatch.chdir(tmp_path)
    _call(y, groups, None, lambda: _compute(y, groups))
    assert not list(tmp_path.iterdir())


def test_rejects_invalid_new_partitions_without_publishing(inputs, tmp_path):
    y, groups = inputs
    splits, seed, score = _compute(y, groups)
    with pytest.raises(ValueError, match="coverage"):
        _call(y, groups, tmp_path, lambda: ((splits[0], splits[0], splits[2]), seed, score))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("operations_per_patient", [
    pytest.param(1, id="single-operation"), pytest.param(2, id="repeat-operations"),
])
def test_validation_cold_and_disk_cache_match_full_64_candidate_oracle(
    inputs, tmp_path, monkeypatch, request, operations_per_patient,
):
    import pandas as pd
    import model_creation.validation as validation
    rng = np.random.default_rng(42)
    n = 48
    y = pd.Series(rng.permutation([0] * 36 + [1] * 12))
    groups = pd.Series(np.arange(n) // operations_per_patient)
    X = pd.DataFrame({"signal": np.arange(n)})
    candidates = [_compute(y.to_numpy(), groups.to_numpy(), 3, seed) for seed in range(42, 106)]
    candidates = [candidate for candidate in candidates
                  if all(y.iloc[train].nunique() == 2 and y.iloc[test].nunique() == 2
                         for train, test in candidate[0])]
    expected = min(candidates, key=lambda candidate: candidate[2])
    monkeypatch.setenv("AKI_SPLIT_CACHE_DIR", str(tmp_path))
    validation._balanced_grouped_splits.cache_clear()
    request.addfinalizer(validation._balanced_grouped_splits.cache_clear)
    cold, _ = validation.build_cv_splits(X, y, groups, 3, 42)
    metadata = validation.describe_cv_split(y, groups, 3, 42)
    _assert_equal((cold, metadata["selected_seed"], metadata["class_count_imbalance"]), expected)
    assert len(list(tmp_path.glob("*.json"))) == 1
    validation._balanced_grouped_splits.cache_clear()
    monkeypatch.setattr(StratifiedGroupKFold, "split", lambda *a, **k: pytest.fail("disk cache reran SGKF"))
    warm, _ = validation.build_cv_splits(X, y, groups, 3, 42)
    _assert_equal((warm, metadata["selected_seed"], metadata["class_count_imbalance"]), expected)
