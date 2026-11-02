"""Saved-artifact canary gates, with independent tiny valid bundle fixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_creation.scheduler_canary import compare_canary


def write_json(path, payload):
    path.write_text(json.dumps(payload))


def refresh_integrity(fold):
    metadata = json.loads((fold / "metadata.json").read_text())
    names = ["model.json", "preprocessor.pkl", "train_oof.csv", "test_predictions.csv", "calibration.json", "threshold.json", "best_params.json"]
    hashes = {name: hashlib.sha256((fold / name).read_bytes()).hexdigest() for name in names}
    metadata.update(artifact_sha256=hashes, model_sha256=hashes["model.json"], preprocessor_sha256=hashes["preprocessor.pkl"])
    write_json(fold / "metadata.json", metadata)


def make_config(root, threads):
    root.mkdir()
    fingerprint = {"checkpoint_bundle_protocol_version": 2, "outer_folds": 2, "inner_folds": 3,
                   "calibration_folds": 5, "repeats": 1, "n_trials": 2, "threads_per_model": threads,
                   "diagnostic": False, "model_type": "xgboost", "validation_scheme": "nested_cv",
                   "outcome": "icu_admission", "branch": "non_windowed", "feature_set": "preop_and_all_waveforms",
                   "cohort_scope": "icu_waveform_sensitivity", "dataset_hash": "fixed", "code_hash": "fixed",
                   "thread_environment": {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": str(threads)},
                   "selected_case_hash": "fixed", "fold_assignment_hash": "fixed"}
    pool = {"train_oof": [], "test": []}
    (root / "artifacts/trials").mkdir(parents=True)
    for number in range(2):
        fold = root / "artifacts/folds/repeat_00" / f"outer_{number:02d}"
        fold.mkdir(parents=True)
        for split, offset in (("test", number * 5), ("train_oof", (1 - number) * 5)):
            raw = np.asarray([.1, .7, .4, .8, .3], dtype=np.float32).astype(float)
            frame = pd.DataFrame({"caseid": np.arange(offset, offset + 5), "subjectid": np.arange(offset, offset + 5) + 100,
                                  "y_true": [0, 1, 0, 1, 0], "y_prob_raw": raw,
                                  "y_prob_calibrated": [.1, .7, .4, .8, .3], "threshold": .5,
                                  "y_pred_label": [0, 1, 0, 1, 0], "fold": number if split == "test" else np.arange(5),
                                  "outer_fold_id": number, "repeat_id": 0})
            frame.to_csv(fold / ("test_predictions.csv" if split == "test" else "train_oof.csv"), index=False)
            pool[split].append(frame)
        write_json(fold / "calibration.json", {"intercept": 0.0, "slope": 1.0, "eps": 1e-6})
        write_json(fold / "threshold.json", {"threshold": .5, "youden_j": 1.0, "sensitivity": 1.0, "specificity": 1.0})
        write_json(fold / "best_params.json", {"max_depth": 3})
        write_json(root / "artifacts/trials" / f"repeat_00_outer_{number:02d}.json", [
            {"number": 0, "params": {"max_depth": 2}, "value": .2, "state": "COMPLETE"},
            {"number": 1, "params": {"max_depth": 3}, "value": .4, "state": "COMPLETE"},
        ])
        (fold / "model.json").write_text("dummy serialized model")
        (fold / "preprocessor.pkl").write_bytes(b"dummy serialized preprocessor")
        write_json(fold / "metadata.json", {"status": "completed", "checkpoint_bundle_protocol_version": 2,
                   "validation_fingerprint": fingerprint, "model_type": "xgboost", "model_path": "model.json",
                   "actual_inner_folds": 3, "actual_calibration_folds": 5, "repeat_id": 0, "outer_fold_id": number,
                   "best_score": .4})
        refresh_integrity(fold)
    (root / "predictions").mkdir()
    for split, frames in pool.items():
        # Production Step 07 attaches these only to pooled exports.
        pd.concat(frames).assign(compute_profile="manuscript", selected_case_hash="fixed").to_csv(
            root / "predictions" / f"{split}.csv", index=False)
    return root


@pytest.fixture
def configs(tmp_path):
    return make_config(tmp_path / "legacy", 12), make_config(tmp_path / "optimized", 4)


def compare(configs, **kwargs):
    return compare_canary(*configs, expected_folds=2, expected_trials=2, n_bootstrap=kwargs.pop("n_bootstrap", 10), **kwargs)


def test_equivalent_reordered_artifacts_pass_without_fitting(configs, monkeypatch):
    from xgboost import XGBClassifier
    monkeypatch.setattr(XGBClassifier, "fit", lambda *a, **kw: pytest.fail("Canary must not fit"))
    for root in configs:
        for path in root.glob("**/*.csv"):
            frame = pd.read_csv(path, float_precision="round_trip")
            frame.iloc[::-1].to_csv(path, index=False)
        for fold in root.glob("artifacts/folds/repeat_*/outer_*"):
            refresh_integrity(fold)
    before = {str(path): path.read_bytes() for root in configs for path in root.rglob("*") if path.is_file()}
    result = compare(configs)
    assert result["status"] == "passed"
    assert result["pooled"]["test"]["rows"] == 10
    assert result["pooled_metrics"]["auroc"] == 1
    assert result["pooled_metrics"]["auprc"] == 1
    assert result["bootstrap"]["draw_hash"]
    assert {str(path): path.read_bytes() for root in configs for path in root.rglob("*") if path.is_file()} == before


def test_trial_parameter_or_trial_score_drift_rejected(configs):
    path = configs[1] / "artifacts/trials/repeat_00_outer_00.json"
    trials = json.loads(path.read_text())
    trials[0]["value"] += 1e-8
    write_json(path, trials)
    with pytest.raises(ValueError, match="trial 0 AP"):
        compare(configs, n_bootstrap=0)
    trials[0]["value"] -= 1e-8
    trials[0]["params"]["max_depth"] = 9
    write_json(path, trials)
    with pytest.raises(ValueError, match="parameter/state sequence"):
        compare(configs, n_bootstrap=0)


@pytest.mark.parametrize("column,value", [("selected_case_hash", "wrong-roster"),
                                          ("compute_profile", "diagnostic"), ("unknown_metadata", "unexpected")])
def test_pooled_annotations_must_match_fingerprint(configs, column, value):
    path = configs[1] / "predictions/test.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame[column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="annotation"):
        compare(configs, n_bootstrap=0)


def test_corrupt_bundle_and_missing_history_rejected(configs):
    fold = configs[1] / "artifacts/folds/repeat_00/outer_00"
    (fold / "model.json").write_text("corrupt")
    with pytest.raises(ValueError, match="Incomplete/corrupt"):
        compare(configs, n_bootstrap=0)
    refresh_integrity(fold)
    (configs[1] / "artifacts/trials/repeat_00_outer_00.json").unlink()
    with pytest.raises(FileNotFoundError):
        compare(configs, n_bootstrap=0)


def test_calibration_and_prediction_drift_rejected(configs):
    fold = configs[1] / "artifacts/folds/repeat_00/outer_00"
    write_json(fold / "calibration.json", {"intercept": 1e-8, "slope": 1, "eps": 1e-6})
    refresh_integrity(fold)
    with pytest.raises(ValueError, match="calibration intercept"):
        compare(configs, n_bootstrap=0)
    write_json(fold / "calibration.json", {"intercept": 0, "slope": 1, "eps": 1e-6})
    path = fold / "test_predictions.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[0, "y_prob_raw"] += 1e-5
    frame.to_csv(path, index=False)
    pooled = configs[1] / "predictions/test.csv"
    root_frame = pd.read_csv(pooled, float_precision="round_trip")
    root_frame.loc[root_frame.caseid.eq(frame.loc[0, "caseid"]), "y_prob_raw"] = frame.loc[0, "y_prob_raw"]
    root_frame.to_csv(pooled, index=False)
    refresh_integrity(fold)
    with pytest.raises(ValueError, match="raw float32 probabilities"):
        compare(configs, n_bootstrap=0)


def test_fold_reduction_or_wrong_source_fingerprint_rejected(configs):
    fold = configs[1] / "artifacts/folds/repeat_00/outer_00"
    path = fold / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["actual_inner_folds"] = 2
    write_json(path, metadata)
    with pytest.raises(ValueError, match="fold budget/identity"):
        compare(configs, n_bootstrap=0)
    metadata["actual_inner_folds"] = 3
    write_json(path, metadata)
    for path in configs[1].glob("artifacts/folds/repeat_*/outer_*/metadata.json"):
        metadata = json.loads(path.read_text())
        metadata["validation_fingerprint"]["dataset_hash"] = "different-data"
        write_json(path, metadata)
    with pytest.raises(ValueError, match="fingerprints differ"):
        compare(configs, n_bootstrap=0)


def test_pooled_union_or_patient_identity_mismatch_rejected(configs):
    path = configs[1] / "predictions/test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "subjectid"] = 999
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="fold union identities"):
        compare(configs, n_bootstrap=0)
