from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import pytest
import psutil
from joblib import delayed

import results_recreation.metrics_summary as metrics_summary
from results_recreation.metrics_summary import (
    PredictionSet,
    _validate_dataframe,
    _generate_bootstrap_indices,
    _load_case_subject_lookup,
    _prediction_frames_are_aligned,
    compute_point_metrics,
    summarize,
)


def _prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4, 5],
            "subjectid": [10, 10, 20, 30, 30],
            "y_true": [0, 1, 0, 1, 0],
            "y_prob_calibrated": [0.2, 0.7, 0.3, 0.8, 0.4],
            "threshold": [0.5, 0.5, 0.5, 0.5, 0.5],
            "y_pred_label": [0, 1, 0, 1, 0],
        }
    )


def test_generate_bootstrap_indices_clusters_subject_rows_together():
    df = _prediction_df()

    samples = _generate_bootstrap_indices(df, n_bootstrap=20, stratified=False, seed=0)

    assert len(samples) == 20
    for sample_idx in samples:
        # Subject 10 contributes rows 0 and 1 together whenever sampled.
        assert np.sum(sample_idx == 0) == np.sum(sample_idx == 1)
        # Subject 30 contributes rows 3 and 4 together whenever sampled.
        assert np.sum(sample_idx == 3) == np.sum(sample_idx == 4)


def test_prediction_frames_are_aligned_checks_case_order_and_subjects():
    reference = _prediction_df()
    same = reference.copy()
    reordered = reference.iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)
    changed_subject = reference.copy()
    changed_subject.loc[0, "subjectid"] = 999

    assert _prediction_frames_are_aligned(reference, same)
    assert not _prediction_frames_are_aligned(reference, reordered)
    assert not _prediction_frames_are_aligned(reference, changed_subject)


def test_prediction_frames_are_aligned_checks_nested_metadata():
    reference = _prediction_df().assign(repeat_id=[0, 0, 0, 1, 1], outer_fold_id=[0, 0, 1, 1, 2])
    same = reference.copy()
    changed_repeat = reference.copy()
    changed_repeat.loc[0, "repeat_id"] = 99

    assert _prediction_frames_are_aligned(reference, same)
    assert not _prediction_frames_are_aligned(reference, changed_repeat)


def test_compute_point_metrics_uses_stored_labels_when_thresholds_vary():
    df = pd.DataFrame(
        {
            "caseid": [1, 2],
            "subjectid": [10, 20],
            "y_true": [1, 0],
            "y_prob_calibrated": [0.6, 0.6],
            "threshold": [0.5, 0.7],
            "y_pred_label": [1, 0],
            "outcome": ["o", "o"],
            "branch": ["b", "b"],
            "feature_set": ["f", "f"],
            "model_name": ["m", "m"],
        }
    )
    pred_set = PredictionSet(
        path=Path("nested.csv"),
        df=df,
        threshold=0.6,
        outcome="o",
        branch="b",
        feature_set="f",
        model_name="m",
        pipeline=None,
    )

    metrics = compute_point_metrics(pred_set)

    assert metrics["threshold"] == pytest.approx(0.6)
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_load_case_subject_lookup_supplements_cohort_from_clinical_file(tmp_path, monkeypatch):
    cohort_path = tmp_path / "cohort.csv"
    clinical_path = tmp_path / "clinical.csv"
    cohort_path.write_text("caseid,subjectid\n1,100\n", encoding="ascii")
    clinical_path.write_text("caseid,subjectid\n1,100\n2,200\n", encoding="ascii")

    monkeypatch.setattr(metrics_summary, "COHORT_FILE", cohort_path)
    monkeypatch.setattr(metrics_summary, "INPUT_FILE", clinical_path)
    _load_case_subject_lookup.cache_clear()
    try:
        lookup = _load_case_subject_lookup()
    finally:
        _load_case_subject_lookup.cache_clear()

    assert lookup is not None
    assert lookup.loc[1] == 100
    assert lookup.loc[2] == 200


def test_validate_dataframe_rejects_duplicate_caseids_in_test_predictions(tmp_path: Path):
    df = _prediction_df().assign(
        y_prob_raw=lambda frame: frame["y_prob_calibrated"],
        fold=0,
        is_oof=True,
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        model_name="XGBoost",
        pipeline="step_07_train_evaluate_nested_cv",
    )
    df.loc[4, "caseid"] = 4

    with pytest.raises(metrics_summary.ValidationError, match="duplicate caseids"):
        _validate_dataframe(df, tmp_path / "test.csv")


def test_summarize_tolerates_empty_bootstrap_frames(monkeypatch):
    df = pd.DataFrame(
        {
            "caseid": [1, 2],
            "subjectid": [10, 20],
            "y_true": [1, 0],
            "y_prob_calibrated": [0.6, 0.4],
            "threshold": [0.5, 0.5],
            "y_pred_label": [1, 0],
            "outcome": ["any_aki", "any_aki"],
            "branch": ["windowed", "windowed"],
            "feature_set": ["all_waveforms", "all_waveforms"],
            "model_name": ["xgboost", "xgboost"],
            "pipeline": ["step_07_train_evaluate_holdout", "step_07_train_evaluate_holdout"],
        }
    )
    pred_set = PredictionSet(
        path=Path("test.csv"),
        df=df,
        threshold=0.5,
        outcome="any_aki",
        branch="windowed",
        feature_set="all_waveforms",
        model_name="xgboost",
        pipeline="step_07_train_evaluate_holdout",
    )

    monkeypatch.setattr(metrics_summary, "_run_bootstrap_jobs", lambda *args, **kwargs: [pd.DataFrame()])

    summary_df, bootstrap_df = summarize([pred_set], n_bootstrap=10, n_jobs=1)

    assert len(summary_df) == 1
    assert summary_df.loc[0, "auroc"] == 1
    assert summary_df.loc[0, "auprc"] == 1
    assert summary_df.loc[0, "brier"] == pytest.approx(.16)
    assert summary_df.loc[0, "auroc_bootstrap_valid"] == 0
    assert pd.isna(summary_df.loc[0, "auroc_lower"])
    assert bootstrap_df is None


def test_native_process_deadline_aborts_workers_without_thread_fallback(tmp_path):
    # Warm the reusable worker pool so this tests cancellation of running tasks,
    # rather than a deadline expiring while Python workers are still importing.
    metrics_summary.Parallel(n_jobs=2, backend="loky")([delayed(abs)(-1), delayed(abs)(-2)])
    def sleeper():
        (tmp_path / f"worker-{os.getpid()}").write_text("started")
        time.sleep(4)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="Bootstrapping failed"):
            metrics_summary._run_bootstrap_jobs(
                [delayed(sleeper)(), delayed(sleeper)()], n_jobs=2,
                parallel_backend="processes", timeout=.3, max_retries=1,
            )
        assert time.monotonic() - started < 2
        pids = [int(path.name.removeprefix("worker-")) for path in tmp_path.glob("worker-*")]
        assert pids and os.getpid() not in pids
        for pid in pids:
            try:
                assert psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                pass
    finally:
        for path in tmp_path.glob("worker-*"):
            pid = int(path.name.removeprefix("worker-"))
            if pid == os.getpid():
                continue
            try:
                process = psutil.Process(pid)
                if process.status() != psutil.STATUS_ZOMBIE:
                    process.kill()
                    process.wait(timeout=2)
            except psutil.NoSuchProcess:
                pass


def test_disabled_deadline_fallback_executes_delayed_tasks_in_original_seed_order(monkeypatch):
    real_parallel = metrics_summary.Parallel
    backends = []
    def parallel(*args, **kwargs):
        backend = kwargs.get("backend", "sequential")
        backends.append(backend)
        if backend != "sequential":
            def failed(tasks):
                raise RuntimeError("unavailable test backend")
            return failed
        return real_parallel(*args, **kwargs)
    monkeypatch.setattr(metrics_summary, "Parallel", parallel)
    seeds = [42, 3, 42]
    def sample(seed):
        return float(np.random.default_rng(seed).normal())
    actual = metrics_summary._run_bootstrap_jobs(
        [delayed(sample)(seed) for seed in seeds], n_jobs=2,
        parallel_backend="processes", timeout=0, max_retries=1,
    )
    assert backends == ["loky", "threading", "sequential"]
    assert actual == [sample(seed) for seed in seeds]


@pytest.mark.parametrize("backend,jobs", [("threads", 2), ("processes", 1)])
def test_uncancellable_execution_warns_and_preserves_explicit_worker_choice(backend, jobs, caplog):
    result = metrics_summary._run_bootstrap_jobs(
        [delayed(sum)([1, 2])], n_jobs=jobs, parallel_backend=backend,
        timeout=1, max_retries=1,
    )
    assert result == [3]
    assert "cannot enforce a hard task deadline" in caplog.text
