"""Saved-artifact equivalence gate for the full-budget XGBoost scheduler canary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_creation.validation import is_outer_fold_checkpoint_complete
from results_recreation import metrics_summary as metrics
from reporting.decision_curve import (
    net_benefit, outcome_thresholds, bootstrap_net_benefit, cluster_resample_counts,
)

PROBABILITY_ATOL = 1e-12
CALIBRATION_ATOL = 1e-10
EXECUTION_FIELDS = {"threads_per_model", "thread_environment"}
KEYS = ["repeat_id", "outer_fold_id", "caseid"]
REQUIRED = {*KEYS, "subjectid", "y_true", "y_prob_raw", "y_prob_calibrated", "threshold", "y_pred_label", "fold"}


def _read(path: Path):
    return json.loads(path.read_text())


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _close(left, right, *, label, atol=PROBABILITY_ATOL, equal_nan=False):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (left.shape != right.shape or np.isinf(left).any() or np.isinf(right).any()
            or not np.allclose(left, right, atol=atol, rtol=0, equal_nan=equal_nan)):
        raise ValueError(f"Canary mismatch: {label} (absolute tolerance {atol})")
    finite = np.isfinite(left) & np.isfinite(right)
    return float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0


def _prediction_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    missing = REQUIRED - set(frame)
    if missing or frame.empty or frame[list(REQUIRED)].isna().any().any():
        raise ValueError(f"Missing/invalid canary prediction fields in {path}: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Duplicate canary operation/fold identity in {path}")
    if not frame.y_true.isin([0, 1]).all() or not frame.y_pred_label.isin([0, 1]).all():
        raise ValueError(f"Nonbinary labels in {path}")
    for column in ("y_prob_raw", "y_prob_calibrated", "threshold"):
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid {column} in {path}")
    if not np.array_equal(frame.y_pred_label, (frame.y_prob_calibrated >= frame.threshold).astype(int)):
        raise ValueError(f"Predicted classes disagree with saved thresholds in {path}")
    return frame.sort_values(KEYS).reset_index(drop=True)


def _compare_predictions(left: pd.DataFrame, right: pd.DataFrame, label: str) -> dict:
    if set(left) != set(right):
        raise ValueError(f"Canary mismatch: {label} prediction columns")
    left = left.sort_values(KEYS).reset_index(drop=True)
    right = right.sort_values(KEYS).reset_index(drop=True)
    numeric = {"y_prob_raw", "y_prob_calibrated", "threshold"}
    columns = sorted(set(left) - numeric)
    try:
        pd.testing.assert_frame_equal(left[columns], right[columns], check_exact=True, check_dtype=False)
    except AssertionError as error:
        raise ValueError(f"Canary mismatch: {label} identities/classes/metadata") from error
    if not np.array_equal(left.y_prob_raw.to_numpy(dtype=np.float32), right.y_prob_raw.to_numpy(dtype=np.float32)):
        raise ValueError(f"Canary mismatch: {label} raw float32 probabilities")
    return {
        "rows": len(left),
        "raw_float32_exact": True,
        "calibrated_max_absolute_difference": _close(left.y_prob_calibrated, right.y_prob_calibrated, label=f"{label} calibrated probabilities"),
        "threshold_max_absolute_difference": _close(left.threshold, right.threshold, label=f"{label} row thresholds"),
    }


def _validate_run(config: Path, *, expected_folds: int, expected_trials: int, expected_threads: int):
    folds = sorted((config / "artifacts/folds").glob("repeat_*/outer_*"))
    expected_names = [("repeat_00", f"outer_{number:02d}") for number in range(expected_folds)]
    if [(path.parent.name, path.name) for path in folds] != expected_names:
        raise ValueError(f"Canary requires exactly the requested outer folds: {config}")
    baseline_fingerprint = None
    fold_frames = {"train_oof": [], "test": []}
    result = []
    hashes = {}
    for number, fold in enumerate(folds):
        if not is_outer_fold_checkpoint_complete(fold):
            raise ValueError(f"Incomplete/corrupt canary fold bundle: {fold}")
        metadata = _read(fold / "metadata.json")
        fingerprint = metadata["validation_fingerprint"]
        expected = {
            "outer_folds": expected_folds, "inner_folds": 3, "calibration_folds": 5,
            "repeats": 1, "n_trials": expected_trials, "threads_per_model": expected_threads,
            "diagnostic": False, "model_type": "xgboost", "validation_scheme": "nested_cv",
            "outcome": "icu_admission", "branch": "non_windowed",
            "feature_set": "preop_and_all_waveforms", "cohort_scope": "icu_waveform_sensitivity",
        }
        for key, value in expected.items():
            if fingerprint.get(key) != value:
                raise ValueError(f"Unexpected canary {key} in {fold}: {fingerprint.get(key)!r}")
        if (metadata.get("actual_inner_folds") != 3 or metadata.get("actual_calibration_folds") != 5
                or metadata.get("repeat_id") != 0 or metadata.get("outer_fold_id") != number):
            raise ValueError(f"Canary fold budget/identity differs from requested protocol: {fold}")
        if baseline_fingerprint is not None and fingerprint != baseline_fingerprint:
            raise ValueError(f"Canary fingerprints differ within one configuration: {fold}")
        baseline_fingerprint = fingerprint
        history_path = config / "artifacts/trials" / f"repeat_00_outer_{number:02d}.json"
        trials = _read(history_path)
        if (len(trials) != expected_trials or [trial.get("number") for trial in trials] != list(range(expected_trials))
                or any(trial.get("state") != "COMPLETE" or not isinstance(trial.get("params"), dict)
                       or not np.isfinite(trial.get("value", np.nan)) for trial in trials)):
            raise ValueError(f"Missing/incomplete canary HPO history: {history_path}")
        best_trial = max(trials, key=lambda trial: trial["value"])
        if best_trial["params"] != _read(fold / "best_params.json"):
            raise ValueError(f"Best parameters disagree with canary trial history: {fold}")
        _close(metadata["best_score"], best_trial["value"], label=f"{fold} best score")
        train = _prediction_frame(fold / "train_oof.csv")
        test = _prediction_frame(fold / "test_predictions.csv")
        if (train.caseid.duplicated().any() or test.caseid.duplicated().any()
                or set(train.caseid) & set(test.caseid) or set(train.subjectid) & set(test.subjectid)):
            raise ValueError(f"Operation/patient leakage within canary fold: {fold}")
        if train.groupby("subjectid").fold.nunique().max() != 1:
            raise ValueError(f"Patient crosses calibration folds in canary: {fold}")
        for frame in (train, test):
            if not frame.repeat_id.eq(0).all() or not frame.outer_fold_id.eq(number).all():
                raise ValueError(f"Prediction fold identity disagrees with canary bundle: {fold}")
        fold_frames["train_oof"].append(train)
        fold_frames["test"].append(test)
        result.append({"fold": fold, "metadata": metadata, "trials": trials, "train": train, "test": test})
        for path in [history_path, *[file for file in fold.iterdir() if file.is_file()]]:
            hashes[str(path.relative_to(config))] = _digest(path)
    pooled = {}
    for split, frames in fold_frames.items():
        path = config / "predictions" / f"{split}.csv"
        pooled[split] = _prediction_frame(path)
        union = pd.concat(frames, ignore_index=True)
        # Step 07 annotates pooled exports after concatenating fold outputs.
        # Check those annotations against the fingerprint before comparing the
        # original fold columns; never discard unknown extra columns silently.
        extra_columns = set(pooled[split]) - set(union)
        allowed_annotations = {
            "compute_profile": baseline_fingerprint.get("compute_profile", "manuscript"),
            "selected_case_hash": baseline_fingerprint["selected_case_hash"],
        }
        if extra_columns - allowed_annotations.keys():
            raise ValueError(f"Unexpected pooled prediction annotations in {path}: {sorted(extra_columns)}")
        for column in extra_columns:
            if not pooled[split][column].eq(allowed_annotations[column]).all():
                raise ValueError(f"Pooled prediction annotation differs from fingerprint: {column}: {path}")
        _compare_predictions(union, pooled[split][union.columns], f"{config}/{split} fold union")
        hashes[str(path.relative_to(config))] = _digest(path)
    test = pooled["test"]
    if test.caseid.duplicated().any() or test.groupby("subjectid").outer_fold_id.nunique().max() != 1:
        raise ValueError(f"Duplicate operation/patient outer-test coverage in canary: {config}")
    universe = set(test.caseid)
    for item in result:
        if set(item["train"].caseid) | set(item["test"].caseid) != universe:
            raise ValueError(f"Canary outer fold does not cover the complete selected population: {item['fold']}")
    return result, pooled, hashes, baseline_fingerprint


def compare_canary(
    legacy_config_dir: Path, optimized_config_dir: Path, *, n_bootstrap: int = 5000,
    bootstrap_seed: int = 0, expected_folds: int = 5, expected_trials: int = 50,
) -> dict:
    """Compare saved full-budget canaries. Raises ValueError on any failed gate.

    Reduced expected fold/trial counts are for unit fixtures only; CLI fixes the
    manuscript 5/3/5 and 50-trial budgets. No estimator or calibrator is fitted.
    """
    if n_bootstrap < 0:
        raise ValueError("Bootstrap count must be nonnegative")
    legacy_config_dir, optimized_config_dir = Path(legacy_config_dir), Path(optimized_config_dir)
    old, old_frames, old_hashes, old_fingerprint = _validate_run(
        legacy_config_dir, expected_folds=expected_folds, expected_trials=expected_trials, expected_threads=12)
    new, new_frames, new_hashes, new_fingerprint = _validate_run(
        optimized_config_dir, expected_folds=expected_folds, expected_trials=expected_trials, expected_threads=4)
    if ({key: value for key, value in old_fingerprint.items() if key not in EXECUTION_FIELDS}
            != {key: value for key, value in new_fingerprint.items() if key not in EXECUTION_FIELDS}):
        raise ValueError("Canary scientific/source/environment fingerprints differ")
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if old_fingerprint["thread_environment"].get(name) != new_fingerprint["thread_environment"].get(name):
            raise ValueError(f"Canary changed BLAS setting: {name}")
    fold_checks = []
    for reference, candidate in zip(old, new):
        fold_label = reference["fold"].name
        for left, right in zip(reference["trials"], candidate["trials"]):
            if {key: value for key, value in left.items() if key != "value"} != {key: value for key, value in right.items() if key != "value"}:
                raise ValueError(f"Canary mismatch: {fold_label} trial parameter/state sequence")
            _close(left["value"], right["value"], label=f"{fold_label} trial {left['number']} AP")
        if _read(reference["fold"] / "best_params.json") != _read(candidate["fold"] / "best_params.json"):
            raise ValueError(f"Canary mismatch: {fold_label} chosen parameters")
        left_cal, right_cal = _read(reference["fold"] / "calibration.json"), _read(candidate["fold"] / "calibration.json")
        if set(left_cal) != set(right_cal):
            raise ValueError(f"Canary mismatch: {fold_label} calibration fields")
        for key in left_cal:
            _close(left_cal[key], right_cal[key], label=f"{fold_label} calibration {key}",
                   atol=CALIBRATION_ATOL if key in {"intercept", "slope"} else PROBABILITY_ATOL)
        left_threshold, right_threshold = _read(reference["fold"] / "threshold.json"), _read(candidate["fold"] / "threshold.json")
        if set(left_threshold) != set(right_threshold):
            raise ValueError(f"Canary mismatch: {fold_label} threshold fields")
        for key in left_threshold:
            _close(left_threshold[key], right_threshold[key], label=f"{fold_label} threshold {key}")
        fold_checks.append({"fold": fold_label, "trials": len(reference["trials"]),
                            "train_oof": _compare_predictions(reference["train"], candidate["train"], f"{fold_label}/train"),
                            "test": _compare_predictions(reference["test"], candidate["test"], f"{fold_label}/test")})
    pooled_checks = {split: _compare_predictions(old_frames[split], new_frames[split], f"pooled/{split}") for split in old_frames}
    prediction_sets = []
    for config, frames in ((legacy_config_dir, old_frames), (optimized_config_dir, new_frames)):
        frame = frames["test"]
        prediction_sets.append(metrics.PredictionSet(config / "predictions/test.csv", frame, float(frame.threshold.median()),
                                                    "icu_admission", "non_windowed", "preop_and_all_waveforms", "xgboost", "catch22",
                                                    "icu_waveform_sensitivity", "nested_cv", "complete"))
    point = [metrics.compute_point_metrics(prediction_set) for prediction_set in prediction_sets]
    for key in point[0]:
        _close(point[0][key], point[1][key], label=f"pooled point metric {key}")
    thresholds = outcome_thresholds("icu_admission")
    dca = [net_benefit(ps.df.y_true, ps.df.y_prob_calibrated, thresholds) for ps in prediction_sets]
    _close(dca[0].to_numpy(), dca[1].to_numpy(), label="pooled DCA")
    bootstrap = {"replicates": n_bootstrap, "seed": bootstrap_seed, "stratified": False}
    if n_bootstrap:
        indices = metrics._generate_bootstrap_indices(prediction_sets[0].df, n_bootstrap, False, bootstrap_seed)
        bootstrap["draw_hash"] = metrics.bootstrap_draw_hash(indices)
        replicates = [metrics._bootstrap_metrics(ps, n_bootstrap, indices_list=indices) for ps in prediction_sets]
        for key in point[0]:
            _close(replicates[0][key], replicates[1][key], label=f"bootstrap metric {key}", equal_nan=True)
            _close(np.nanquantile(replicates[0][key], [.025, .975]), np.nanquantile(replicates[1][key], [.025, .975]),
                   label=f"bootstrap interval {key}", equal_nan=True)
        counts = cluster_resample_counts(prediction_sets[0].df.subjectid.to_numpy(), indices)
        curves = [bootstrap_net_benefit(ps.df.y_true, ps.df.y_prob_calibrated, thresholds,
                                       groups=ps.df.subjectid.to_numpy(), cluster_counts=counts) for ps in prediction_sets]
        _close(curves[0][1], curves[1][1], label="bootstrap DCA replicates")
    return {"status": "passed", "legacy_config": str(legacy_config_dir.resolve()), "optimized_config": str(optimized_config_dir.resolve()),
            "tolerances": {"probability_metrics_absolute": PROBABILITY_ATOL, "calibration_coefficients_absolute": CALIBRATION_ATOL,
                           "raw_probabilities": "exact after float32 restoration", "relative": 0},
            "folds": fold_checks, "pooled": pooled_checks, "pooled_metrics": point[0], "bootstrap": bootstrap,
            "legacy_fingerprint": old_fingerprint, "optimized_fingerprint": new_fingerprint,
            "artifact_sha256": {"legacy": old_hashes, "optimized": new_hashes}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-config", type=Path, required=True)
    parser.add_argument("--optimized-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    try:
        payload = compare_canary(args.legacy_config, args.optimized_config, n_bootstrap=args.bootstrap, bootstrap_seed=args.bootstrap_seed)
    except (ValueError, OSError, KeyError, TypeError) as error:
        payload = {"status": "failed", "error": str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        raise SystemExit(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
