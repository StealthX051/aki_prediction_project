"""Read-only verification of complete configurations and the protected core study.

The legacy adapter is deliberately pinned to one reviewed source revision. It
does not relax checkpoint resume rules or modify any source artifact.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path

import pandas as pd

from data_preparation.artifact_metadata import file_sha256
from data_preparation.cohort_scopes import select_cohort, selected_case_hash
from model_creation.validation import (
    compare_validation_fingerprints,
    is_outer_fold_checkpoint_complete,
)

LEGACY_REVISION = "fcff7c7c9794e0ea8405814c68176760a2eb9afc"
LEGACY_CODE_HASH = "0592413369b4380af27f785a1c0481c8418a3bf1aceca829452beee9069b789e"
# Modeling-source digest reviewed at f3b5dcf; consumer-only commits can change
# revision while retaining this digest. A later modeling change needs review.
REVIEWED_EXTENSION_CODE_HASH = "926b0f7a5eb8710b3d4de7a88ecab8ddad3a43de52e18a3c574e599ec3d30ce0"
LEGACY_ENVIRONMENT = {
    "scikit-learn": "1.6.1", "numpy": "2.0.2", "pandas": "2.2.3",
    "xgboost-cpu": "2.1.4", "interpret-core": "0.7.8", "optuna": "4.6.0",
}
IDENTITY_FIELDS = ("cohort_scope", "model_type", "outcome", "branch", "feature_set")
SOURCE_NAMES = {
    "non_windowed": "aki_features_master_wide.csv",
    "windowed": "aki_features_master_wide_windowed.csv",
}
TARGETS = {"any_aki": "aki_label", "icu_admission": "y_icu_admit", "mortality": "y_inhosp_mortality"}
EXTENSION_OUTCOMES = {"icu_waveform_sensitivity": "icu_admission",
                      "mortality_waveform_sensitivity": "mortality"}
EXTENSION_ADAPTER = "reviewed-extension-core-supplement-v1"


def configuration_key(config: dict, scope: str | None = None) -> str:
    """Stable full logical identity; identical names in another scope never alias."""
    values = dict(config)
    if scope is not None:
        values["cohort_scope"] = scope
    return "/".join(str(values[field]) for field in IDENTITY_FIELDS)


def validate_reviewed_extension(target_identity: dict) -> None:
    """Authorize reuse only with the specifically reviewed modeling extension."""
    if target_identity.get("code_hash") != REVIEWED_EXTENSION_CODE_HASH:
        raise ValueError("Protected reuse requires the reviewed modeling code hash; unreviewed modeling changes need a new compatibility review")
    expected = {"environment": LEGACY_ENVIRONMENT, "validation_protocol_version": 4,
                "preprocessing_protocol": "fold-local-v4-finite-numeric-schema"}
    for key, value in expected.items():
        if target_identity.get(key) != value:
            raise ValueError(f"Scientific protocol differs across reused/new runs: {key}")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _same_frame(left: pd.DataFrame, right: pd.DataFrame, columns: list[str], context: str) -> None:
    keys = [column for column in ("repeat_id", "outer_fold_id", "caseid") if column in columns]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(f"Duplicate prediction identity: {context}")
    try:
        pd.testing.assert_frame_equal(
            left[columns].sort_values(keys).reset_index(drop=True),
            right[columns].sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError(f"Prediction identity/value mismatch: {context}") from exc


def verify_completed_configuration(
    config_dir: Path,
    *,
    expected_fingerprint: dict | None = None,
    dataset_path: Path | None = None,
    dataset: pd.DataFrame | None = None,
    legacy: bool = False,
) -> dict:
    """Verify bundles, pooled values, patient separation and original provenance.

    Callers skipping a *current* run must pass its freshly computed fingerprint;
    this function never declares arbitrary source revisions interchangeable.
    Dataset dataframes may be supplied to avoid reading the same CSV 18 times;
    the file's SHA256 is still checked on every call.
    """
    config_dir = Path(config_dir).resolve()
    artifacts = config_dir / "artifacts"
    validation_path = artifacts / "validation.json"
    validation = _read_json(validation_path)
    fp = validation["validation_fingerprint"]
    if expected_fingerprint is not None:
        matches, reason = compare_validation_fingerprints(expected_fingerprint, fp)
        if not matches:
            raise ValueError(f"Completed configuration fingerprint mismatch: {reason}")
    if fp.get("validation_scheme") != "nested_cv":
        raise ValueError("Completed inventory requires nested-CV bundles")
    if legacy:
        family = fp.get("model_type")
        blas = "4" if family == "logistic_regression" else "1"
        required = {
            "code_revision": LEGACY_REVISION, "code_hash": LEGACY_CODE_HASH,
            "environment": LEGACY_ENVIRONMENT, "validation_protocol_version": 4,
            "checkpoint_bundle_protocol_version": 2,
            "preprocessing_protocol": "fold-local-v4-finite-numeric-schema",
            "cohort_scope": "main_shared", "outer_folds": 5, "inner_folds": 3,
            "calibration_folds": 5, "repeats": 1, "diagnostic": False,
            "base_random_state": 42, "threads_per_model": 12,
            "n_trials": 20 if fp.get("model_type") == "logistic_regression" else 50,
            "preserve_nan": True, "legacy_imputation": False, "compute_profile": "manuscript",
            "thread_environment": {"OPENBLAS_NUM_THREADS": blas, "MKL_NUM_THREADS": blas,
                                   "OMP_NUM_THREADS": "12", "NUMBA_NUM_THREADS": "12"},
            "class_weighting": {"logistic_regression": "none", "xgboost": "fold_training_scale_pos_weight_only",
                                "ebm": "fold_training_sample_weight_only"}.get(family),
        }
        for key, value in required.items():
            if fp.get(key) != value:
                raise ValueError(f"Unapproved legacy {key}: {fp.get(key)!r}")
        for package, version in LEGACY_ENVIRONMENT.items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Legacy replay environment mismatch: {package}")

    fold_dirs = sorted((artifacts / "folds").glob("repeat_*/outer_*"))
    expected_pairs = set(itertools.product(range(fp["repeats"]), range(fp["outer_folds"])))
    fold_manifest_path = artifacts / "folds_manifest.json"
    entries = _read_json(fold_manifest_path)["folds"]
    manifest_pairs = [(row["repeat_id"], row["outer_fold_id"]) for row in entries]
    if len(manifest_pairs) != len(expected_pairs) or set(manifest_pairs) != expected_pairs:
        raise ValueError("Incomplete or duplicate fold manifest")
    if any(row.get("status") not in {"completed", "resumed"} for row in entries):
        raise ValueError("Fold manifest contains unfinished work")
    tests, trains, fold_rows, checksums = [], [], {}, {}
    checksums[str(validation_path)] = file_sha256(validation_path)
    checksums[str(fold_manifest_path)] = file_sha256(fold_manifest_path)
    root_metadata = _read_json(artifacts / "metadata.json")
    if root_metadata.get("validation_fingerprint") != fp:
        raise ValueError("Pooled metadata fingerprint differs from validation manifest")
    for name in ("metadata.json", "calibration.json", "threshold.json"):
        path = artifacts / name
        checksums[str(path)] = file_sha256(path)
    for fold in fold_dirs:
        if not is_outer_fold_checkpoint_complete(fold):
            raise ValueError(f"Incomplete/corrupt fold bundle: {fold}")
        metadata = _read_json(fold / "metadata.json")
        matches, reason = compare_validation_fingerprints(fp, metadata.get("validation_fingerprint"))
        if not matches:
            raise ValueError(f"Fold fingerprint mismatch: {fold}: {reason}")
        if not fp.get("diagnostic"):
            for name, value in (("actual_inner_folds", fp["inner_folds"]),
                                ("requested_inner_folds", fp["inner_folds"]),
                                ("actual_calibration_folds", fp["calibration_folds"]),
                                ("requested_calibration_folds", fp["calibration_folds"]),
                                ("n_trials", fp["n_trials"])):
                if metadata.get(name) != value:
                    raise ValueError(f"Reduced or incomplete definitive fold setting: {name}: {fold}")
        pair = (metadata["repeat_id"], metadata["outer_fold_id"])
        if pair in fold_rows:
            raise ValueError("Duplicate fold bundle identity")
        train = pd.read_csv(fold / "train_oof.csv")
        test = pd.read_csv(fold / "test_predictions.csv")
        for frame in (train, test):
            if frame["caseid"].duplicated().any() or frame[["caseid", "subjectid", "y_true"]].isna().any().any():
                raise ValueError(f"Invalid fold operation/patient identities: {fold}")
            if not frame["y_true"].isin([0, 1]).all():
                raise ValueError(f"Invalid fold truth: {fold}")
            if not frame["y_pred_label"].isin([0, 1]).all():
                raise ValueError(f"Invalid fold predicted label: {fold}")
            for column in ("y_prob_raw", "y_prob_calibrated", "threshold"):
                if not pd.to_numeric(frame[column], errors="coerce").between(0, 1).all():
                    raise ValueError(f"Invalid fold probability/threshold: {column}: {fold}")
            if not fp.get("diagnostic") and frame["y_true"].nunique() != 2:
                raise ValueError(f"Definitive fold must contain both outcome classes: {fold}")
            if not frame["repeat_id"].eq(pair[0]).all() or not frame["outer_fold_id"].eq(pair[1]).all():
                raise ValueError(f"Prediction fold identity mismatch: {fold}")
        if set(train.subjectid) & set(test.subjectid):
            raise ValueError(f"Patient leakage between outer train/test: {fold}")
        fold_rows[pair] = (train, test)
        tests.append(test)
        trains.append(train)
        checksums[str(fold / "metadata.json")] = file_sha256(fold / "metadata.json")
        checksums.update({str(fold / name): digest for name, digest in metadata["artifact_sha256"].items()})
    if set(fold_rows) != expected_pairs:
        raise ValueError("Incomplete outer-fold bundles")
    pooled_path = config_dir / "predictions" / "test.csv"
    pooled = pd.read_csv(pooled_path)
    merged = pd.concat(tests, ignore_index=True)
    _same_frame(pooled, merged, list(merged.columns), str(pooled_path))
    checksums[str(pooled_path)] = file_sha256(pooled_path)
    train_path = config_dir / "predictions" / "train_oof.csv"
    pooled_train = pd.read_csv(train_path)
    merged_train = pd.concat(trains, ignore_index=True)
    _same_frame(pooled_train, merged_train, list(merged_train.columns), str(train_path))
    checksums[str(train_path)] = file_sha256(train_path)
    if selected_case_hash(pooled.loc[pooled.repeat_id.eq(0), ["caseid", "subjectid"]]) != fp["selected_case_hash"]:
        raise ValueError("Selected operation/patient hash mismatch")
    for _, frame in pooled.groupby("repeat_id"):
        if frame.groupby("subjectid")["outer_fold_id"].nunique().gt(1).any():
            raise ValueError("Patient crosses outer-test folds")

    if dataset_path is not None:
        dataset_path = Path(dataset_path).resolve()
        if file_sha256(dataset_path) != fp["dataset_hash"]:
            raise ValueError(f"Modeling input changed: {dataset_path}")
        selected = select_cohort(pd.read_csv(dataset_path) if dataset is None else dataset,
                                 fp["cohort_scope"], fp["outcome"])
        if selected_case_hash(selected) != fp["selected_case_hash"]:
            raise ValueError("Modeling input roster mismatch")
        truth = selected[["caseid", "subjectid", TARGETS[fp["outcome"]]]].rename(columns={TARGETS[fp["outcome"]]: "y_true"})
        roster = []
        for repeat in range(fp["repeats"]):
            _same_frame(pooled.loc[pooled.repeat_id.eq(repeat)], truth, ["caseid", "subjectid", "y_true"], "source truth")
            for outer in range(fp["outer_folds"]):
                train, test = fold_rows[repeat, outer]
                source_test = selected.loc[selected.caseid.isin(test.caseid)]
                source_train = selected.loc[~selected.caseid.isin(test.caseid)]
                _same_frame(train, truth.loc[truth.caseid.isin(source_train.caseid)], ["caseid", "subjectid", "y_true"], "source training complement")
                if repeat == 0:
                    roster.append({"train": source_train.caseid.tolist(), "test": source_test.caseid.tolist()})
        digest = hashlib.sha256(json.dumps(roster, sort_keys=True).encode()).hexdigest()
        if digest != fp["fold_assignment_hash"]:
            raise ValueError("Source operation fold-assignment hash mismatch")
        if file_sha256(dataset_path) != fp["dataset_hash"]:
            raise ValueError(f"Modeling input changed during verification: {dataset_path}")
        checksums[str(dataset_path)] = fp["dataset_hash"]

    return {
        "key": configuration_key(fp), "configuration_dir": str(config_dir),
        "prediction_path": str(pooled_path), "validation_fingerprint": fp,
        "fold_count": len(fold_rows), "operations": len(pooled),
        "artifact_sha256": checksums, "status": "verified_completed",
    }


def verify_legacy_reference(config: dict, attestation: dict, target_identity: dict) -> dict:
    """Check a reporting reference without treating old fingerprints as current."""
    if (attestation.get("adapter") != "reviewed-fcff7c7-scheduler-scope-extension-v1"
            or attestation.get("target_study_identity") != target_identity):
        raise ValueError("Missing approved source-to-target compatibility attestation")
    source_identity = attestation.get("source_study_identity", {})
    required_source = {
        "code_revision": LEGACY_REVISION, "code_hash": LEGACY_CODE_HASH,
        "environment": LEGACY_ENVIRONMENT, "validation_protocol_version": 4,
        "preprocessing_protocol": "fold-local-v4-finite-numeric-schema",
    }
    if source_identity != required_source:
        raise ValueError("Compatibility attestation has an unapproved source protocol")
    for key in ("environment", "validation_protocol_version", "preprocessing_protocol"):
        if target_identity.get(key) != source_identity[key]:
            raise ValueError(f"Scientific protocol differs across reused/new runs: {key}")
    validate_reviewed_extension(target_identity)
    reference = config.get("reuse_reference", {})
    directory = Path(config["artifact_path"]).resolve()
    if str(directory) != reference.get("configuration_dir"):
        raise ValueError("External reference directory differs from imported inventory")
    input_paths = [Path(path) for path in reference.get("artifact_sha256", {})
                   if Path(path).name == SOURCE_NAMES.get(config["branch"])]
    if len(input_paths) != 1:
        raise ValueError("External reference is missing its verified modeling input")
    record = verify_completed_configuration(directory, legacy=True, dataset_path=input_paths[0])
    if record != reference:
        raise ValueError("External reference changed since inventory import")
    if record["key"] != configuration_key(config, "main_shared"):
        raise ValueError("External reference logical identity differs from requested configuration")
    return record


def import_legacy_inventory(source: Path, *, processed_dir: Path | None = None) -> dict:
    """Return a verified reference inventory; never create files in either root."""
    source = Path(source).resolve()
    results = source if (source / "run_manifest.json").is_file() else source / "results/catch22/experiments/main_shared/nested_cv"
    manifest_path = results / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "validated" or manifest.get("failures"):
        raise ValueError("Protected baseline is not a validated complete study")
    identity = manifest.get("study_identity", {})
    if identity.get("code_revision") != LEGACY_REVISION or identity.get("code_hash") != LEGACY_CODE_HASH:
        raise ValueError("Only the reviewed fcff7c7 baseline can be imported")
    expected = set(itertools.product(
        ("logistic_regression", "xgboost", "ebm"), ("any_aki", "icu_admission"),
        ("non_windowed", "windowed"), ("preop_only", "all_waveforms", "preop_and_all_waveforms"),
    ))
    fields = ("model_type", "outcome", "branch", "feature_set")
    configured = [tuple(config[key] for key in fields) for config in manifest["configurations"]]
    if len(configured) != 36 or set(configured) != expected:
        raise ValueError("Protected baseline must contain exactly the approved 36 core configurations")
    processed = Path(processed_dir or manifest["processed_dir"]).resolve()
    datasets = {branch: pd.read_csv(processed / name) for branch, name in SOURCE_NAMES.items()}
    inventory = {}
    fold_hashes = {}
    for config in manifest["configurations"]:
        record = verify_completed_configuration(
            results.joinpath("models", *(config[key] for key in fields)),
            dataset_path=processed / SOURCE_NAMES[config["branch"]],
            dataset=datasets[config["branch"]], legacy=True,
        )
        fp = record["validation_fingerprint"]
        for key in ("selected_case_hash", "dataset_sha256"):
            fingerprint_key = "dataset_hash" if key == "dataset_sha256" else key
            if manifest["branch_rosters"][config["branch"]][key] != fp[fingerprint_key]:
                raise ValueError("Protected manifest branch roster differs from model input")
        group = (fp["outcome"], fp["branch"])
        if fold_hashes.setdefault(group, fp["fold_assignment_hash"]) != fp["fold_assignment_hash"]:
            raise ValueError("Protected models did not share outcome/branch folds")
        inventory[record["key"]] = record
    return {
        "inventory_version": 1, "source_results_dir": str(results),
        "source_manifest_sha256": file_sha256(manifest_path), "configurations": inventory,
        "compatibility_attestation": {
            "adapter": "reviewed-fcff7c7-scheduler-scope-extension-v1",
            "source_study_identity": identity, "source_preserved_read_only": True,
            "validation": "original bundle hashes, source truth/rosters, patient separation, fold assignment and pooled fold union",
            "allowed_changes": ["configuration scheduling", "resource admission", "additional outcome scopes"],
            "does_not_authorize": ["checkpoint fingerprint relaxation", "cross-cohort fitted-model reuse"],
        },
    }


def _validate_extension_fingerprint(fp: dict, identity: dict) -> None:
    """Keep reviewed extension budgets, preprocessing and numerical settings fixed."""
    validate_reviewed_extension(identity)
    scope, family = fp.get("cohort_scope"), fp.get("model_type")
    if scope not in EXTENSION_OUTCOMES or family not in {"logistic_regression", "xgboost", "ebm"}:
        raise ValueError("Unsupported completed extension scope or model family")
    native, blas = (4 if family == "xgboost" else 12), (4 if family == "logistic_regression" else 1)
    required = {
        **identity, "outcome": EXTENSION_OUTCOMES[scope],
        "validation_scheme": "nested_cv", "diagnostic": False,
        "outer_folds": 5, "inner_folds": 3, "calibration_folds": 5, "repeats": 1,
        "checkpoint_bundle_protocol_version": 2, "base_random_state": 42,
        "n_trials": 20 if family == "logistic_regression" else 50,
        "preserve_nan": True, "legacy_imputation": False, "compute_profile": "manuscript",
        "threads_per_model": native,
        "thread_environment": {"OMP_NUM_THREADS": str(native), "NUMBA_NUM_THREADS": str(native),
                               "OPENBLAS_NUM_THREADS": str(blas), "MKL_NUM_THREADS": str(blas)},
    }
    if not identity.get("code_revision"):
        raise ValueError("Completed extension is missing its original source revision")
    for field, value in required.items():
        if fp.get(field) != value:
            raise ValueError(f"Unapproved completed extension setting: {field}")


def _verify_extension_config(config: dict, identity: dict) -> None:
    reference = config.get("reuse_reference") or config.get("completed_reference", {})
    fp = reference.get("validation_fingerprint", {})
    _validate_extension_fingerprint(fp, identity)
    if reference.get("key") != configuration_key(config) or configuration_key(fp) != configuration_key(config):
        raise ValueError("Completed extension reference logical identity differs from configuration")
    if config.get("n_trials") != fp["n_trials"]:
        raise ValueError("Completed extension HPO budget differs from requested configuration")
    resources = config.get("resources", {})
    if (resources.get("threads_per_model") != fp["threads_per_model"]
            or str(resources.get("blas_threads")) != fp["thread_environment"]["OPENBLAS_NUM_THREADS"]):
        raise ValueError("Completed extension numerical resources differ from requested configuration")


def validate_completed_reference_request(config: dict, attestation: dict) -> None:
    """Reject incompatible modern reuse requests before queue writes or new fits."""
    if attestation.get("adapter") == EXTENSION_ADAPTER:
        _verify_extension_config(config, attestation.get("source_study_identity", {}))


def _import_extension_inventory(results: Path, processed_dir: Path | None) -> dict:
    manifest_path = results / "run_manifest.json"
    manifest_hash = file_sha256(manifest_path)
    manifest = _read_json(manifest_path)
    scope = manifest.get("cohort_scope")
    if scope not in EXTENSION_OUTCOMES:
        raise ValueError("Only reviewed broader ICU/mortality core scopes can be imported")
    if manifest.get("status") != "validated" or manifest.get("failures"):
        raise ValueError("Completed extension is not a validated complete study")
    for field, value in {"validation_scheme": "nested_cv", "diagnostic": False,
                         "outer_folds": 5, "inner_folds": 3, "calibration_folds": 5, "repeats": 1}.items():
        if manifest.get(field) != value:
            raise ValueError(f"Completed extension manifest has unapproved {field}")
    identity = manifest.get("study_identity", {})
    validate_reviewed_extension(identity)
    expected = set(itertools.product(
        (scope,), ("logistic_regression", "xgboost", "ebm"), (EXTENSION_OUTCOMES[scope],),
        ("non_windowed", "windowed"), ("preop_only", "all_waveforms", "preop_and_all_waveforms")))
    configs = manifest.get("configurations", [])
    if len(configs) != 18 or {tuple(c.get(field) for field in IDENTITY_FIELDS) for c in configs} != expected:
        raise ValueError("Completed extension must contain exactly the approved 18 core configurations")
    plan = manifest.get("split_plan", {})
    if plan.get("status") != "validated":
        raise ValueError("Completed extension is missing a validated frozen split plan")
    processed = Path(processed_dir or manifest["processed_dir"]).resolve()
    datasets = {branch: pd.read_csv(processed / name) for branch, name in SOURCE_NAMES.items()}
    inventory = {}
    for config in configs:
        _verify_extension_config(config, identity)
        reference = config["completed_reference"]
        directory = results.joinpath("models", *(config[field] for field in IDENTITY_FIELDS[1:])).resolve()
        if reference.get("configuration_dir") != str(directory):
            raise ValueError("Completed extension reference points outside its source configuration")
        record = verify_completed_configuration(
            directory, expected_fingerprint=reference["validation_fingerprint"],
            dataset_path=processed / SOURCE_NAMES[config["branch"]], dataset=datasets[config["branch"]])
        if record != reference:
            raise ValueError("Completed extension reference changed since source validation")
        fp = record["validation_fingerprint"]
        roster = manifest.get("branch_rosters", {}).get(config["branch"], {})
        if (roster.get("dataset_sha256") != fp["dataset_hash"]
                or roster.get("selected_case_hash") != fp["selected_case_hash"]):
            raise ValueError("Completed extension branch roster differs from model input")
        split_key = "/".join(config[field] for field in ("cohort_scope", "outcome", "branch"))
        if plan.get("fold_assignment_hashes", {}).get(split_key) != fp["fold_assignment_hash"]:
            raise ValueError("Completed extension models differ from frozen outcome/branch folds")
        inventory[record["key"]] = record
    if file_sha256(manifest_path) != manifest_hash:
        raise ValueError("Completed extension manifest changed during import")
    return {
        "inventory_version": 1, "cohort_scope": scope, "source_results_dir": str(results),
        "source_manifest_sha256": manifest_hash,
        "source_split_cache_dir": plan.get("cache_dir"), "configurations": inventory,
        "compatibility_attestation": {
            "adapter": EXTENSION_ADAPTER, "source_study_identity": identity,
            "source_preserved_read_only": True,
            "validation": "original fingerprints and bundle hashes, source truth/rosters, patient separation and frozen fold assignment",
            "allowed_changes": ["configuration scheduling", "resource admission", "additional feature comparisons"],
            "does_not_authorize": ["checkpoint fingerprint relaxation", "cross-cohort fitted-model reuse"],
        },
    }


def import_completed_inventory(source: Path, *, processed_dir: Path | None = None,
                               scopes: list[str] | None = None) -> list[dict]:
    """Import scoped completed cores read-only; preserve the legacy main adapter."""
    source = Path(source).resolve()
    if (source / "run_manifest.json").is_file():
        manifest = _read_json(source / "run_manifest.json")
        if manifest.get("study_identity", {}).get("code_hash") == LEGACY_CODE_HASH:
            return [import_legacy_inventory(source, processed_dir=processed_dir)]
        if scopes is not None and manifest.get("cohort_scope") not in scopes:
            raise ValueError("Completed source scope is not among requested scopes")
        return [_import_extension_inventory(source, processed_dir)]
    base = source / "results/catch22/experiments"
    main = base / "main_shared/nested_cv"
    extension = [base / scope / "nested_cv" for scope in EXTENSION_OUTCOMES
                 if (base / scope / "nested_cv/run_manifest.json").is_file()]
    if (main / "run_manifest.json").is_file():
        if extension:
            raise ValueError("Mixed main/extension source roots must be imported separately")
        # Legacy protection remains checked even if this request fits only new scopes.
        return [import_legacy_inventory(source, processed_dir=processed_dir)]
    selected = [path for path in extension if scopes is None or path.parent.name in scopes]
    if not selected:
        raise ValueError("No completed source manifests for the requested scopes")
    return [_import_extension_inventory(path, processed_dir) for path in selected]


def verify_completed_reference(config: dict, attestation: dict, target_identity: dict) -> dict:
    """Dispatch source-specific reuse without rewriting original fingerprints."""
    if attestation.get("adapter") != EXTENSION_ADAPTER:
        return verify_legacy_reference(config, attestation, target_identity)
    if attestation.get("target_study_identity") != target_identity:
        raise ValueError("Missing approved source-to-target compatibility attestation")
    identity = attestation.get("source_study_identity", {})
    validate_reviewed_extension(identity)
    validate_reviewed_extension(target_identity)
    validate_completed_reference_request(config, attestation)
    reference = config["reuse_reference"]
    directory = Path(config["artifact_path"]).resolve()
    if str(directory) != reference.get("configuration_dir"):
        raise ValueError("External reference directory differs from imported inventory")
    inputs = [Path(path) for path in reference.get("artifact_sha256", {})
              if Path(path).name == SOURCE_NAMES.get(config["branch"])]
    if len(inputs) != 1:
        raise ValueError("External reference is missing its verified modeling input")
    record = verify_completed_configuration(
        directory, expected_fingerprint=reference["validation_fingerprint"], dataset_path=inputs[0])
    if record != reference:
        raise ValueError("External reference changed since inventory import")
    return record
