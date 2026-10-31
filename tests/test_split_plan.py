"""Split warming preserves exact partitions and validates actual EBM bags."""
import numpy as np
import pandas as pd
import pytest

from model_creation import split_plan, validation


@pytest.fixture
def repeated_patients(monkeypatch, request, tmp_path):
    monkeypatch.setenv("AKI_SPLIT_CACHE_DIR", str(tmp_path / "initial-cache"))
    # Each outcome is stable within a patient, so class support must come from
    # different patients rather than merely repeated rows.
    groups = pd.Series(np.repeat(np.arange(24), 2))
    truth = pd.Series(np.repeat([0] * 16 + [1] * 8, 2))
    validation._balanced_grouped_splits.cache_clear()
    request.addfinalizer(validation._balanced_grouped_splits.cache_clear)
    return truth, groups


def test_warmed_partitions_and_cache_bytes_match_serial_selection(tmp_path, monkeypatch, repeated_patients):
    truth, groups = repeated_patients
    serial, warmed = tmp_path / "serial", tmp_path / "warmed"
    monkeypatch.setenv("AKI_SPLIT_CACHE_DIR", str(serial))
    expected, actual = validation.build_cv_splits(pd.DataFrame(index=truth.index), truth, groups, 3, 42)
    assert actual == 3
    # Clear the in-memory memo so warming must independently select partitions.
    validation._balanced_grouped_splits.cache_clear()
    got = split_plan._partitions((truth.tolist(), groups.tolist(), 3, 42, str(warmed), False))
    assert got == [(train.tolist(), test.tolist()) for train, test in expected]
    assert {p.name: p.read_bytes() for p in serial.glob("*.json")} == {
        p.name: p.read_bytes() for p in warmed.glob("*.json")}
    for train, test in got:
        assert not set(groups.iloc[train]) & set(groups.iloc[test])
        assert truth.iloc[train].nunique() == truth.iloc[test].nunique() == 2
    # A warm cache must not invoke the grouped splitter again.
    validation._balanced_grouped_splits.cache_clear()
    monkeypatch.setattr(validation.StratifiedGroupKFold, "split",
                        lambda *a, **k: pytest.fail("Valid warm cache recomputed partitions"))
    assert split_plan._partitions((truth.tolist(), groups.tolist(), 3, 42, str(warmed), False)) == got


def test_preflight_rejects_silent_effective_fold_reduction(tmp_path, monkeypatch):
    monkeypatch.setenv("AKI_SPLIT_CACHE_DIR", str(tmp_path))
    validation._balanced_grouped_splits.cache_clear()
    with pytest.raises(ValueError, match="refusing silent reduction"):
        split_plan._partitions(([0, 0, 0, 0, 1, 1], list(range(6)), 5, 42, str(tmp_path), False))


def test_preflight_checks_production_bags_on_full_and_every_inner_training_set(
    tmp_path, monkeypatch, repeated_patients,
):
    truth, groups = repeated_patients
    original = validation.make_ebm_bags
    checked = []
    def inspect_bags(y, patients, **kwargs):
        bags = original(y, patients, **kwargs)
        assert bags.shape == (len(y), 14)
        for bag in bags.T:
            training = np.flatnonzero(bag > 0)
            testing = np.flatnonzero(bag < 0)
            assert not set(patients.iloc[training]) & set(patients.iloc[testing])
            assert y.iloc[training].nunique() == y.iloc[testing].nunique() == 2
        checked.append((tuple(patients.tolist()), kwargs["validation_size"]))
        return bags
    monkeypatch.setattr(validation, "make_ebm_bags", inspect_bags)
    splits = split_plan._partitions((truth.tolist(), groups.tolist(), 3, 42, str(tmp_path), True))
    expected_patients = [tuple(groups), *(tuple(groups.iloc[train]) for train, _ in splits)]
    assert sorted(checked) == sorted((patients, size) for patients in expected_patients for size in (.1, .15, .2))


def test_preflight_propagates_infeasible_bag_failure(tmp_path, monkeypatch, repeated_patients):
    truth, groups = repeated_patients
    def impossible(*args, **kwargs):
        raise ValueError("Cannot make patient-grouped EBM internal partitions with both classes")
    monkeypatch.setattr(validation, "make_ebm_bags", impossible)
    with pytest.raises(ValueError, match="both classes"):
        split_plan._partitions((truth.tolist(), groups.tolist(), 3, 42, str(tmp_path), True))


def test_spawned_preflight_warms_exact_outer_hpo_and_calibration_partitions(tmp_path, monkeypatch):
    import os
    cache = tmp_path / "cache"
    processed = tmp_path / "processed"
    processed.mkdir()
    groups = np.repeat(np.arange(30), 2)
    mortality = np.repeat([0] * 20 + [1] * 10, 2)
    icu = np.repeat([0, 1] * 15, 2)
    frame = pd.DataFrame({"caseid": np.arange(60), "subjectid": groups, "age": 55,
        "eligible_mortality": 1, "y_inhosp_mortality": mortality,
        "eligible_icu_admission": 1, "y_icu_admit": icu})
    configurations = []
    for branch, suffix in (("non_windowed", ""), ("windowed", "_windowed")):
        frame.to_csv(processed / f"aki_features_master_wide{suffix}.csv", index=False)
        for scope, outcome in (("icu_waveform_sensitivity", "icu_admission"),
                               ("mortality_waveform_sensitivity", "mortality")):
            configurations.append({"cohort_scope": scope, "outcome": outcome,
                                   "branch": branch, "model_type": "xgboost"})
    names = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    plan = split_plan.prepare_split_plan(processed, configurations, cache, workers=2)
    assert plan["status"] == "validated"
    assert plan["unique_outer_keys"] == 2  # matching branches share partitions
    assert plan["nested_requests"] == 20
    assert {name: os.environ.get(name) for name in names} == previous
    monkeypatch.setenv("AKI_SPLIT_CACHE_DIR", str(cache))
    validation._balanced_grouped_splits.cache_clear()
    monkeypatch.setattr(validation.StratifiedGroupKFold, "split",
                        lambda *a, **k: pytest.fail("Preflight missed a required partition cache"))
    try:
        for values in (mortality, icu):
            truth, patients = pd.Series(values), pd.Series(groups)
            outer, actual = validation.build_cv_splits(pd.DataFrame(index=truth.index), truth, patients, 5, 42)
            assert actual == 5
            for fold, (train, test) in enumerate(outer):
                assert not set(patients.iloc[train]) & set(patients.iloc[test])
                for count in (3, 5):
                    nested, effective = validation.build_cv_splits(
                        pd.DataFrame(index=np.arange(len(train))), truth.iloc[train], patients.iloc[train], count, 42 + fold)
                    assert effective == count
                    for inner_train, inner_test in nested:
                        assert not set(patients.iloc[train].iloc[inner_train]) & set(patients.iloc[train].iloc[inner_test])
    finally:
        validation._balanced_grouped_splits.cache_clear()
