"""Validated, non-pickle reuse of deterministic patient-grouped partitions."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _canonical(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _validate(result, outcomes, patient_ids, n_splits, random_state):
    splits, seed, score = result
    y, groups = np.asarray(outcomes), np.asarray(patient_ids)
    if (len(y) != len(groups) or len(splits) != n_splits or np.unique(y).size != 2
            or not isinstance(seed, (int, np.integer)) or isinstance(seed, bool)
            or not random_state <= seed < random_state + 64):
        raise ValueError("Split cache has incompatible lengths, fold count, or seed")
    _, encoded = np.unique(y, return_inverse=True)
    target = np.bincount(encoded) / n_splits
    seen, calculated_score = [], 0.0
    for train, test in splits:
        for indices in (train, test):
            if indices.dtype.kind not in "iu" or indices.ndim != 1 or not len(indices):
                raise ValueError("Split indices must be nonempty integer vectors")
            if (indices < 0).any() or (indices >= len(y)).any() or (np.diff(indices) <= 0).any():
                raise ValueError("Split indices are outside range, duplicated, or unordered")
        if not np.array_equal(train, np.setdiff1d(np.arange(len(y)), test)):
            raise ValueError("Training indices must be exactly the test complement")
        if set(groups[train]) & set(groups[test]) or np.unique(y[train]).size != 2 or np.unique(y[test]).size != 2:
            raise ValueError("Split cache violates patient separation or class support")
        seen.extend(test.tolist())
        calculated_score += np.square((np.bincount(encoded[test], minlength=len(target)) - target) / target).sum()
    if sorted(seen) != list(range(len(y))) or score != calculated_score:
        raise ValueError("Split cache has incomplete coverage or an invalid balance score")
    return tuple((train.copy(), test.copy()) for train, test in splits), int(seed), float(score)


def cached_grouped_splits(outcomes, patient_ids, n_splits, random_state, *, policy, compute, cache_dir=None):
    """Read a validated exact-input cache, or compute and atomically publish it."""
    cache_dir = cache_dir or os.getenv("AKI_SPLIT_CACHE_DIR")
    if not cache_dir:
        return compute()
    from artifact_paths import enforce_storage_policy
    root = Path(cache_dir).expanduser().resolve()
    enforce_storage_policy({"split_cache_dir": root})
    identity = {"outcomes": np.asarray(outcomes).tolist(), "patient_ids": np.asarray(patient_ids).tolist(),
                "folds": n_splits, "seed": random_state, "policy": policy,
                "numpy": np.__version__, "scikit-learn": importlib.metadata.version("scikit-learn"),
                "cache_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    key = hashlib.sha256(_canonical(identity)).hexdigest()
    path = root / f"{key}.json"
    try:
        envelope = json.loads(path.read_text())
        payload = envelope["payload"]
        if envelope["sha256"] != hashlib.sha256(_canonical(payload)).hexdigest() or payload["key"] != key:
            raise ValueError("Split-cache checksum or identity mismatch")
        splits = tuple((np.setdiff1d(np.arange(len(outcomes)), test), np.asarray(test))
                       for test in payload["test_indices"])
        result = _validate((splits, payload["selected_seed"], payload["score"]),
                           outcomes, patient_ids, n_splits, random_state)
        logger.info("Reusing validated grouped CV split cache %s", path)
        return result
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Ignoring invalid split cache %s: %s", path, exc)
    result = _validate(compute(), outcomes, patient_ids, n_splits, random_state)
    splits, seed, score = result
    payload = {"key": key, "selected_seed": seed, "score": score,
               "test_indices": [test.tolist() for _, test in splits]}
    envelope = {"payload": payload, "sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
    root.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=f"{key}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(envelope, handle, sort_keys=True)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
