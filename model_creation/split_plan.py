"""Warm exact-input grouped partition caches before configuration admission."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path


def _partitions(task):
    import numpy as np
    import pandas as pd
    from model_creation.validation import build_cv_splits, make_ebm_bags
    y_values, group_values, folds, seed, cache_dir, check_bags = task
    os.environ["AKI_SPLIT_CACHE_DIR"] = str(cache_dir)
    y, groups = pd.Series(y_values), pd.Series(group_values)
    splits, actual = build_cv_splits(pd.DataFrame(index=np.arange(len(y))), y, groups, folds, seed)
    if actual != folds:
        raise ValueError("Requested grouped fold count is infeasible; refusing silent reduction")
    for train, test in splits:
        if y.iloc[train].nunique() != 2 or y.iloc[test].nunique() != 2:
            raise ValueError("Every requested partition must contain both outcome classes")
    if check_bags:
        # HPO's single bag is the first production bag for the same seed.
        for indices in [np.arange(len(y)), *(train for train, _ in splits)]:
            for fraction in (0.1, 0.15, 0.2):
                make_ebm_bags(y.iloc[indices], groups.iloc[indices], n_bags=14,
                              validation_size=fraction, random_state=seed)
    return [(train.tolist(), test.tolist()) for train, test in splits]


def prepare_split_plan(processed_dir: Path, configurations: list[dict], cache_dir: Path,
                       *, outer_folds=5, inner_folds=3, calibration_folds=5,
                       workers=8) -> dict:
    """Preserve selection/tie ordering; parallelize only independent cache keys."""
    import pandas as pd
    from model_creation.validation import select_modeling_dataset
    arrays, case_ids = {}, {}
    for config in configurations:
        key = (config["cohort_scope"], config["outcome"], config["branch"])
        if key not in arrays:
            suffix = "_windowed" if key[2] == "windowed" else ""
            frame = pd.read_csv(processed_dir / f"aki_features_master_wide{suffix}.csv")
            _, _, y, cases, groups = select_modeling_dataset(
                frame, key[1], "preop_only", cohort_scope=key[0])
            arrays[key] = (y.tolist(), groups.tolist())
            case_ids[key] = cases.tolist()
    check_bags = any(c["model_type"] == "ebm" for c in configurations)
    unique = {}
    for key, (y, groups) in arrays.items():
        digest = hashlib.sha256(json.dumps([y, groups]).encode()).hexdigest()
        unique.setdefault(digest, (y, groups))
    # Spawn imports numerical libraries with one thread, avoiding nested BLAS pools.
    names = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            outer_tasks = [(y, g, outer_folds, 42, str(cache_dir), False) for y, g in unique.values()]
            outer_results = list(pool.map(_partitions, outer_tasks))
            by_input = dict(zip(unique, outer_results))
            fold_hashes = {}
            for key, (y, groups) in arrays.items():
                digest = hashlib.sha256(json.dumps([y, groups]).encode()).hexdigest()
                cases = case_ids[key]
                roster = [{"train": [cases[i] for i in train], "test": [cases[i] for i in test]}
                          for train, test in by_input[digest]]
                fold_hashes["/".join(key)] = hashlib.sha256(json.dumps(roster, sort_keys=True).encode()).hexdigest()
            nested_tasks = []
            for (y, groups), outer in zip(unique.values(), outer_results):
                for fold, (train, _) in enumerate(outer):
                    for count in dict.fromkeys((inner_folds, calibration_folds)):
                        nested_tasks.append(([y[i] for i in train], [groups[i] for i in train],
                                             count, 42 + fold, str(cache_dir), check_bags))
            list(pool.map(_partitions, nested_tasks))
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return {"status": "validated", "scope_outcome_branches": [list(k) for k in arrays],
            "unique_outer_keys": len(unique), "nested_requests": len(nested_tasks),
            "outer_folds": outer_folds, "inner_folds": inner_folds,
            "calibration_folds": calibration_folds, "ebm_bags_checked": check_bags,
            "fold_assignment_hashes": fold_hashes,
            "cache_dir": str(cache_dir)}
