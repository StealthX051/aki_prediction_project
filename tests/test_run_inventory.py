"""Completed-run reuse verifies scientific identities without fitting or writes."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from data_preparation.artifact_metadata import file_sha256
from data_preparation.cohort_scopes import selected_case_hash
from model_creation.run_inventory import (
    LEGACY_CODE_HASH, LEGACY_ENVIRONMENT, LEGACY_REVISION, REVIEWED_EXTENSION_CODE_HASH,
    configuration_key, import_completed_inventory, import_legacy_inventory,
    validate_completed_reference_request, verify_completed_configuration,
    verify_completed_reference, verify_legacy_reference,
)


def _write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def complete_run(tmp_path):
    directory = tmp_path / "configuration"
    artifacts = directory / "artifacts"
    predictions = directory / "predictions"
    predictions.mkdir(parents=True)
    source = pd.DataFrame({
        "caseid": [1, 2, 3, 4, 5, 6, 7, 8],
        "subjectid": [10, 10, 20, 30, 40, 40, 50, 60],
        "eligible_any_aki": 1, "aki_label": [0, 0, 1, 0, 1, 1, 0, 1],
        "eligible_icu_admission": 1, "y_icu_admit": [0, 0, 1, 0, 1, 1, 0, 1],
    })
    source_path = tmp_path / "aki_features_master_wide.csv"
    source.to_csv(source_path, index=False)
    partitions = [(source.iloc[4:], source.iloc[:4]), (source.iloc[:4], source.iloc[4:])]
    roster = [{"train": train.caseid.tolist(), "test": test.caseid.tolist()} for train, test in partitions]
    fp = {
        "checkpoint_bundle_protocol_version": 2, "validation_scheme": "nested_cv",
        "model_type": "xgboost", "outcome": "icu_admission", "branch": "non_windowed",
        "feature_set": "preop_only", "cohort_scope": "main_shared", "outer_folds": 2,
        "inner_folds": 2, "calibration_folds": 2, "repeats": 1,
        "diagnostic": True, "selected_case_hash": selected_case_hash(source),
        "dataset_hash": file_sha256(source_path),
        "fold_assignment_hash": hashlib.sha256(json.dumps(roster, sort_keys=True).encode()).hexdigest(),
    }
    trains, tests, manifest = [], [], []
    for index, (train, test) in enumerate(partitions):
        fold = artifacts / "folds" / "repeat_00" / f"outer_{index:02d}"
        fold.mkdir(parents=True)
        for label, frame, records in (("train_oof", train, trains), ("test_predictions", test, tests)):
            frame = frame[["caseid", "subjectid", "y_icu_admit"]].rename(columns={"y_icu_admit": "y_true"})
            frame = frame.assign(repeat_id=0, outer_fold_id=index, y_prob_raw=0.2,
                                 y_prob_calibrated=0.25, y_pred_label=0, threshold=0.5)
            frame.to_csv(fold / f"{label}.csv", index=False)
            records.append(frame)
        for name in ("model.json", "preprocessor.pkl", "calibration.json", "threshold.json", "best_params.json"):
            (fold / name).write_text("{}")
        hashes = {p.name: file_sha256(p) for p in fold.iterdir()}
        _write_json(fold / "metadata.json", {
            "status": "completed", "checkpoint_bundle_protocol_version": 2,
            "validation_fingerprint": fp, "model_type": "xgboost", "model_path": "model.json",
            "model_sha256": hashes["model.json"], "preprocessor_sha256": hashes["preprocessor.pkl"],
            "artifact_sha256": hashes, "repeat_id": 0, "outer_fold_id": index,
        })
        manifest.append({"repeat_id": 0, "outer_fold_id": index, "status": "completed"})
    pd.concat(tests).to_csv(predictions / "test.csv", index=False)
    pd.concat(trains).to_csv(predictions / "train_oof.csv", index=False)
    _write_json(artifacts / "validation.json", {"validation_fingerprint": fp})
    _write_json(artifacts / "folds_manifest.json", {"folds": manifest})
    _write_json(artifacts / "metadata.json", {"validation_fingerprint": fp})
    _write_json(artifacts / "calibration.json", {})
    _write_json(artifacts / "threshold.json", {})
    return directory, source_path, source, fp


def test_complete_verification_is_read_only_and_aligns_reordered_rows(complete_run):
    directory, source_path, _, fp = complete_run
    path = directory / "predictions/test.csv"
    pd.read_csv(path).sample(frac=1, random_state=2).to_csv(path, index=False)
    before = {p: file_sha256(p) for p in directory.rglob("*") if p.is_file()}
    record = verify_completed_configuration(directory, dataset_path=source_path, expected_fingerprint=fp)
    assert record["fold_count"] == 2
    assert record["operations"] == 8
    assert before == {p: file_sha256(p) for p in before}
    assert configuration_key(fp) != configuration_key(fp, "icu_waveform_sensitivity")


def test_different_fingerprint_or_changed_pooled_prediction_fails(complete_run):
    directory, _, _, fp = complete_run
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_completed_configuration(directory, expected_fingerprint={**fp, "outer_folds": 5})
    path = directory / "predictions/test.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "y_prob_calibrated"] = 0.9
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Prediction identity/value mismatch"):
        verify_completed_configuration(directory)


def test_corrupt_or_unfinished_bundle_cannot_be_imported(complete_run):
    directory, _, _, _ = complete_run
    model = next(directory.glob("artifacts/folds/**/model.json"))
    model.write_text("corruption")
    with pytest.raises(ValueError, match="corrupt fold bundle"):
        verify_completed_configuration(directory)


def test_resumed_manifest_requires_the_same_verified_fold_bundles(complete_run):
    directory, source_path, _, _ = complete_run
    path = directory / "artifacts/folds_manifest.json"
    manifest = json.loads(path.read_text())
    for entry in manifest["folds"]:
        entry["status"] = "resumed"
    _write_json(path, manifest)
    assert verify_completed_configuration(directory, dataset_path=source_path)["fold_count"] == 2
    next(directory.glob("artifacts/folds/**/model.json")).unlink()
    with pytest.raises(ValueError, match="corrupt fold bundle"):
        verify_completed_configuration(directory, dataset_path=source_path)


def test_source_truth_and_dataset_hash_are_checked(complete_run):
    directory, path, source, _ = complete_run
    modified = source.copy()
    modified.loc[0, "y_icu_admit"] = 1
    with pytest.raises(ValueError, match="source truth"):
        verify_completed_configuration(directory, dataset_path=path, dataset=modified)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Modeling input changed"):
        verify_completed_configuration(directory, dataset_path=path)


def test_train_patient_cannot_appear_in_outer_test(complete_run):
    directory, _, _, _ = complete_run
    fold = next(directory.glob("artifacts/folds/repeat_*/outer_*"))
    path = fold / "train_oof.csv"
    train = pd.read_csv(path)
    test = pd.read_csv(fold / "test_predictions.csv")
    train.loc[0, "subjectid"] = test.subjectid.iloc[0]
    train.to_csv(path, index=False)
    metadata = json.loads((fold / "metadata.json").read_text())
    metadata["artifact_sha256"][path.name] = file_sha256(path)
    _write_json(fold / "metadata.json", metadata)
    with pytest.raises(ValueError, match="Patient leakage"):
        verify_completed_configuration(directory)


def test_arbitrary_source_revision_has_no_compatibility_exception(tmp_path, complete_run):
    directory, _, _, _ = complete_run
    with pytest.raises(ValueError, match="Unapproved legacy"):
        verify_completed_configuration(directory, legacy=True)
    _write_json(tmp_path / "run_manifest.json", {
        "status": "validated", "study_identity": {"code_revision": "arbitrary", "code_hash": LEGACY_CODE_HASH},
    })
    with pytest.raises(ValueError, match="reviewed fcff7c7"):
        import_legacy_inventory(tmp_path)
    with pytest.raises(ValueError, match="source-to-target"):
        verify_legacy_reference({}, {}, {})


def test_compatibility_attestation_does_not_permit_changed_environment():
    source = {"code_revision": LEGACY_REVISION, "code_hash": LEGACY_CODE_HASH,
              "environment": LEGACY_ENVIRONMENT, "validation_protocol_version": 4,
              "preprocessing_protocol": "fold-local-v4-finite-numeric-schema"}
    target = {**source, "code_revision": "new", "environment": {"numpy": "changed"}}
    attestation = {"adapter": "reviewed-fcff7c7-scheduler-scope-extension-v1",
                   "source_study_identity": source, "target_study_identity": target}
    with pytest.raises(ValueError, match="Scientific protocol differs.*environment"):
        verify_legacy_reference({}, attestation, target)


def test_external_reference_requires_unchanged_inventory_and_exact_target(tmp_path, monkeypatch):
    import model_creation.run_inventory as inventory
    source = {"code_revision": LEGACY_REVISION, "code_hash": LEGACY_CODE_HASH,
              "environment": LEGACY_ENVIRONMENT, "validation_protocol_version": 4,
              "preprocessing_protocol": "fold-local-v4-finite-numeric-schema"}
    target = {**source, "code_revision": "new", "code_hash": REVIEWED_EXTENSION_CODE_HASH}
    attestation = {"adapter": "reviewed-fcff7c7-scheduler-scope-extension-v1",
                   "source_study_identity": source, "target_study_identity": target}
    config = {"model_type": "xgboost", "outcome": "icu_admission", "branch": "non_windowed",
              "feature_set": "preop_only", "artifact_path": str(tmp_path)}
    record = {"key": configuration_key(config, "main_shared"), "configuration_dir": str(tmp_path),
              "artifact_sha256": {str(tmp_path / "aki_features_master_wide.csv"): "source-hash"}}
    config["reuse_reference"] = record
    monkeypatch.setattr(inventory, "verify_completed_configuration", lambda *args, **kwargs: dict(record))
    assert verify_legacy_reference(config, attestation, target) == record
    with pytest.raises(ValueError, match="source-to-target"):
        verify_legacy_reference(config, attestation, {**target, "code_revision": "another"})
    monkeypatch.setattr(inventory, "verify_completed_configuration", lambda *args, **kwargs: {**record, "changed": True})
    with pytest.raises(ValueError, match="changed since inventory"):
        verify_legacy_reference(config, attestation, target)


def test_same_protocol_cannot_attest_arbitrary_modeling_changes():
    source = {"code_revision": LEGACY_REVISION, "code_hash": LEGACY_CODE_HASH,
              "environment": LEGACY_ENVIRONMENT, "validation_protocol_version": 4,
              "preprocessing_protocol": "fold-local-v4-finite-numeric-schema"}
    target = {**source, "code_revision": "unreviewed", "code_hash": "arbitrary-model-code"}
    attestation = {"adapter": "reviewed-fcff7c7-scheduler-scope-extension-v1",
                   "source_study_identity": source, "target_study_identity": target}
    with pytest.raises(ValueError, match="reviewed modeling code hash"):
        verify_legacy_reference({}, attestation, target)


def test_main_reuse_rejects_unreviewed_modeling_before_writes_or_fits(tmp_path, monkeypatch, capsys):
    import run_catch22
    import model_creation.run_inventory as inventory
    root = tmp_path / "untouched"
    for variable, suffix in (("AKI_ARTIFACT_ROOT", "artifacts"), ("PROCESSED_DIR", "processed"),
                             ("RESULTS_DIR", "results"), ("PAPER_DIR", "paper"), ("LOG_FILE", "log.txt")):
        monkeypatch.setenv(variable, str(root / suffix))
    key = "main_shared/xgboost/any_aki/non_windowed/preop_only"
    catalog = {"source_results_dir": str(tmp_path / "protected/results/catch22/experiments/main_shared/nested_cv"),
               "configurations": {key: {"validation_fingerprint": {"n_trials": 50}}}}
    monkeypatch.setattr(inventory, "import_completed_inventory", lambda *a, **kw: [catalog])
    identities = []
    def unreviewed_identity():
        identities.append(True)
        return {"code_hash": "unreviewed"}
    monkeypatch.setattr(run_catch22, "_study_identity", unreviewed_identity)
    monkeypatch.setattr(run_catch22, "run_module", lambda *a, **kw: pytest.fail("Unreviewed source launched work"))
    result = run_catch22.main(["experiments", "--preset", "core", "--model-types", "xgboost",
        "--outcomes", "any_aki", "--branches", "non_windowed", "--feature-sets", "preop_only",
        "--prep", "skip", "--reuse-from", str(tmp_path / "protected"), "--dry-run"])
    assert result == 1
    assert "reviewed modeling code hash" in capsys.readouterr().err
    assert identities == [True]
    assert not root.exists()


def test_reporting_discovers_references_without_copying_artifacts(tmp_path, monkeypatch):
    from results_recreation import metrics_summary as metrics
    source = tmp_path / "protected" / "configuration"
    destination = tmp_path / "new-results"
    destination.mkdir()
    manifest = destination / "run_manifest.json"
    _write_json(manifest, {"configurations": [{"artifact_path": str(source), "reuse_reference": {"verified": True}}]})
    monkeypatch.setattr(metrics, "_load_predictions", lambda path, **kwargs: path)
    assert metrics.crawl_predictions(destination) == [source / "predictions/test.csv"]
    assert not source.exists()
    _write_json(manifest, {"configurations": [{"artifact_path": str(source)}]})
    with pytest.raises(metrics.ValidationError, match="verified reuse reference"):
        metrics.crawl_predictions(destination)


def test_protected_output_guard_handles_scoped_reuse_and_symlinks(tmp_path):
    from types import SimpleNamespace
    from model_creation.experiment_queue import _assert_protected_paths
    protected = tmp_path / "old-study"
    protected.mkdir()
    source = protected / "results/catch22/experiments/main_shared/nested_cv"
    catalog = {"source_results_dir": str(source)}
    alias = tmp_path / "alias"
    alias.symlink_to(protected, target_is_directory=True)
    layout = SimpleNamespace(artifact_root=tmp_path / "new", results_dir=alias / "new-results",
                             paper_dir=tmp_path / "paper")
    contexts = {"main_shared": {"layout": layout, "log": tmp_path / "log"}}
    with pytest.raises(ValueError, match="protected reuse study"):
        _assert_protected_paths(contexts, [catalog])
    layout.results_dir = tmp_path / "new/results"
    _assert_protected_paths(contexts, [catalog])


def test_current_skip_requires_exact_expected_protocol_and_frozen_plan(complete_run, monkeypatch):
    import shutil
    from types import SimpleNamespace
    from model_creation.experiment_queue import _completed_reference
    from model_creation import validation
    directory, source_path, _, fp = complete_run
    results = directory.parent / "results"
    destination = results / "models/xgboost/icu_admission/non_windowed/preop_only"
    shutil.copytree(directory, destination)
    config = {key: fp[key] for key in ("model_type", "outcome", "branch", "feature_set", "cohort_scope")}
    config.update(n_trials=3, resources={"threads_per_model": 2, "blas_threads": 2})
    args = SimpleNamespace(validation_scheme="nested_cv", outer_folds=2, inner_folds=2,
                           calibration_folds=2, repeats=1)
    context = {"layout": SimpleNamespace(results_dir=results, processed_dir=source_path.parent),
               "path": results / "manifest.json", "manifest": {"diagnostic": True}}
    saved = {"split_plan": {"fold_assignment_hashes": {"main_shared/icu_admission/non_windowed": fp["fold_assignment_hash"]}},
             "branch_rosters": {"non_windowed": {"dataset_sha256": fp["dataset_hash"],
                                                  "selected_case_hash": fp["selected_case_hash"]}}}
    # Check all fields from the newly constructed expected fingerprint, not a
    # hand-selected subset of settings; the fixture only needs to store its
    # original minimal fingerprint for this dispatch regression.
    expected = {**fp, "thread_environment": {"OMP_NUM_THREADS": "2", "NUMBA_NUM_THREADS": "2",
                                             "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}}
    monkeypatch.setattr(validation, "build_validation_fingerprint", lambda **kwargs: expected)
    with pytest.raises(ValueError, match="missing keys"):
        _completed_reference(args, config, context, saved)
    with pytest.raises(ValueError, match="frozen fold-assignment"):
        _completed_reference(args, config, context, {})
    # Request merging must not erase evidence that an artifact was previously
    # complete, causing a deleted file to trigger an unapproved fresh fit.
    _write_json(context["path"], {"configurations": [{**config, "completed_reference": {"verified": True}}]})
    (destination / "artifacts/validation.json").unlink()
    with pytest.raises(ValueError, match="lost required artifacts"):
        _completed_reference(args, config, context, {"configurations": [config]})


def test_current_skip_rejects_valid_completed_run_on_input_replaced_after_freeze(complete_run):
    import shutil
    from types import SimpleNamespace
    from model_creation.experiment_queue import _completed_reference
    directory, source_path, source, fp = complete_run
    results = directory.parent / "results"
    destination = results / "models/xgboost/icu_admission/non_windowed/preop_only"
    shutil.copytree(directory, destination)
    saved = {"split_plan": {"fold_assignment_hashes": {"main_shared/icu_admission/non_windowed": fp["fold_assignment_hash"]}},
             "branch_rosters": {"non_windowed": {"dataset_sha256": fp["dataset_hash"],
                                                  "selected_case_hash": fp["selected_case_hash"]}}}
    # A feature-only replacement leaves case, patient, truth and fold identities
    # intact. A self-consistent completed bundle for this different input must
    # still be rejected by the original study freeze.
    source.assign(age=range(len(source))).to_csv(source_path, index=False)
    changed_hash = file_sha256(source_path)
    for path in [destination / "artifacts/validation.json", destination / "artifacts/metadata.json",
                 *destination.glob("artifacts/folds/**/metadata.json")]:
        metadata = json.loads(path.read_text())
        metadata["validation_fingerprint"]["dataset_hash"] = changed_hash
        _write_json(path, metadata)
    assert verify_completed_configuration(destination, dataset_path=source_path)["status"] == "verified_completed"
    config = {key: fp[key] for key in ("model_type", "outcome", "branch", "feature_set", "cohort_scope")}
    args = SimpleNamespace(validation_scheme="nested_cv", outer_folds=2, inner_folds=2,
                           calibration_folds=2, repeats=1)
    context = {"layout": SimpleNamespace(results_dir=results, processed_dir=source_path.parent),
               "path": results / "manifest.json", "manifest": {"diagnostic": True}}
    with pytest.raises(ValueError, match="differs from frozen study manifest"):
        _completed_reference(args, config, context, saved)


@pytest.fixture
def extension_inventory(tmp_path, monkeypatch):
    """Synthetic manifests exercise adapters; bundle integrity has real tests above."""
    import copy
    import itertools
    import model_creation.run_inventory as inventory
    root = tmp_path / "completed"
    processed = tmp_path / "processed"
    processed.mkdir()
    data = pd.DataFrame({"caseid": [1, 2], "subjectid": [10, 20]})
    for filename in inventory.SOURCE_NAMES.values():
        data.to_csv(processed / filename, index=False)
    identity = {"code_hash": REVIEWED_EXTENSION_CODE_HASH, "code_revision": "original-revision",
                "environment": LEGACY_ENVIRONMENT, "validation_protocol_version": 4,
                "preprocessing_protocol": "fold-local-v4-finite-numeric-schema"}
    records, paths = {}, {}
    for scope, outcome in inventory.EXTENSION_OUTCOMES.items():
        results = root / "results/catch22/experiments" / scope / "nested_cv"
        results.mkdir(parents=True)
        configs, rosters, hashes = [], {}, {}
        for family, branch, features in itertools.product(
                ["logistic_regression", "xgboost", "ebm"], ["non_windowed", "windowed"],
                ["preop_only", "all_waveforms", "preop_and_all_waveforms"]):
            native, blas = (4 if family == "xgboost" else 12), (4 if family == "logistic_regression" else 1)
            dataset = processed / inventory.SOURCE_NAMES[branch]
            config = dict(cohort_scope=scope, outcome=outcome, model_type=family,
                          branch=branch, feature_set=features, n_trials=20 if family == "logistic_regression" else 50,
                          resources={"threads_per_model": native, "blas_threads": blas, "max_workers": 5})
            fp = {**identity, **{k: config[k] for k in inventory.IDENTITY_FIELDS},
                  "validation_scheme": "nested_cv", "outer_folds": 5, "inner_folds": 3,
                  "calibration_folds": 5, "repeats": 1, "diagnostic": False,
                  "checkpoint_bundle_protocol_version": 2, "base_random_state": 42,
                  "n_trials": config["n_trials"], "threads_per_model": native,
                  "preserve_nan": True, "legacy_imputation": False, "compute_profile": "manuscript",
                  "thread_environment": {"OMP_NUM_THREADS": str(native), "NUMBA_NUM_THREADS": str(native),
                                         "OPENBLAS_NUM_THREADS": str(blas), "MKL_NUM_THREADS": str(blas)},
                  "dataset_hash": file_sha256(dataset), "selected_case_hash": selected_case_hash(data),
                  "fold_assignment_hash": f"{scope}-{branch}-frozen-folds"}
            directory = results / "models" / family / outcome / branch / features
            record = {"key": configuration_key(config), "configuration_dir": str(directory),
                      "validation_fingerprint": fp, "status": "verified_completed",
                      "artifact_sha256": {str(dataset): fp["dataset_hash"]}}
            records[str(directory)] = copy.deepcopy(record)
            config["completed_reference"] = record
            configs.append(config)
            rosters[branch] = {"dataset_sha256": fp["dataset_hash"], "selected_case_hash": fp["selected_case_hash"]}
            hashes[f"{scope}/{outcome}/{branch}"] = fp["fold_assignment_hash"]
        manifest = {**{k: fp[k] for k in ["validation_scheme", "outer_folds", "inner_folds", "calibration_folds", "repeats", "diagnostic"]},
                    "cohort_scope": scope, "status": "validated", "study_identity": identity,
                    "processed_dir": str(processed), "branch_rosters": rosters, "configurations": configs,
                    "split_plan": {"status": "validated", "cache_dir": str(root / "control/split_cache"),
                                   "fold_assignment_hashes": hashes}}
        paths[scope] = results / "run_manifest.json"
        _write_json(paths[scope], manifest)
    def verify(directory, *, expected_fingerprint=None, dataset_path=None, dataset=None, **kwargs):
        record = copy.deepcopy(records[str(directory)])
        assert expected_fingerprint == record["validation_fingerprint"]
        assert file_sha256(dataset_path) == record["validation_fingerprint"]["dataset_hash"]
        return record
    monkeypatch.setattr(inventory, "verify_completed_configuration", verify)
    return root, paths, records, identity


def test_extension_import_preserves_scopes_originals_and_source_revisions(extension_inventory):
    root, paths, records, _ = extension_inventory
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    catalogs = import_completed_inventory(root)
    assert [len(c["configurations"]) for c in catalogs] == [18, 18]
    assert {k: v for c in catalogs for k, v in c["configurations"].items()} == {v["key"]: v for v in records.values()}
    assert all(c["source_split_cache_dir"] == str(root / "control/split_cache") for c in catalogs)
    assert before == {path: path.read_bytes() for path in before}
    scope = "mortality_waveform_sensitivity"
    assert len(import_completed_inventory(root, scopes=[scope])) == 1
    assert import_completed_inventory(paths[scope].parent, scopes=[scope])[0]["cohort_scope"] == scope


@pytest.mark.parametrize("change, message", [
    ("status", "validated complete"), ("diagnostic", "unapproved diagnostic"),
    ("code_hash", "reviewed modeling code hash"), ("missing_config", "18 core"),
    ("roster", "branch roster"), ("split", "frozen outcome/branch folds"),
    ("budget", "HPO budget"), ("resources", "numerical resources"),
    ("scope", "18 core"), ("source_revision", "setting: code_revision"),
])
def test_extension_import_rejects_changed_science_or_manifest(extension_inventory, change, message):
    root, paths, _, _ = extension_inventory
    path = paths["icu_waveform_sensitivity"]
    manifest = json.loads(path.read_text())
    first = manifest["configurations"][0]
    if change == "status": manifest["status"] = "running"
    elif change == "diagnostic": manifest["diagnostic"] = True
    elif change == "code_hash": manifest["study_identity"]["code_hash"] = "unreviewed"
    elif change == "missing_config": manifest["configurations"].pop()
    elif change == "roster": manifest["branch_rosters"]["non_windowed"]["selected_case_hash"] = "changed"
    elif change == "split": manifest["split_plan"]["fold_assignment_hashes"] = {}
    elif change == "budget": first["n_trials"] = 1
    elif change == "resources": first["resources"]["blas_threads"] = 1
    elif change == "scope": first["cohort_scope"] = "main_shared"
    elif change == "source_revision": manifest["study_identity"]["code_revision"] = "changed"
    _write_json(path, manifest)
    with pytest.raises(ValueError, match=message):
        import_completed_inventory(root, scopes=["icu_waveform_sensitivity"])


def test_extension_reference_requires_scoped_identity_and_unchanged_hashes(extension_inventory, monkeypatch):
    import copy
    import model_creation.run_inventory as inventory
    root, paths, _, identity = extension_inventory
    catalog = import_completed_inventory(root, scopes=["icu_waveform_sensitivity"])[0]
    config = copy.deepcopy(json.loads(paths["icu_waveform_sensitivity"].read_text())["configurations"][0])
    config["reuse_reference"] = config.pop("completed_reference")
    config["artifact_path"] = config["reuse_reference"]["configuration_dir"]
    target = {**identity, "code_revision": "new-consumer-revision"}
    attestation = {**catalog["compatibility_attestation"], "target_study_identity": target}
    validate_completed_reference_request(config, attestation)
    with pytest.raises(ValueError, match="numerical resources"):
        validate_completed_reference_request({**config, "resources": {**config["resources"], "blas_threads": 1}}, attestation)
    assert verify_completed_reference(config, attestation, target) == config["reuse_reference"]
    assert config["reuse_reference"]["validation_fingerprint"]["code_revision"] == "original-revision"
    with pytest.raises(ValueError, match="source-to-target"):
        verify_completed_reference(config, attestation, {**target, "code_revision": "another"})
    with pytest.raises(ValueError, match="logical identity"):
        verify_completed_reference({**config, "cohort_scope": "mortality_waveform_sensitivity"}, attestation, target)
    changed = copy.deepcopy(config["reuse_reference"])
    changed["artifact_sha256"]["a-saved-model"] = "changed"
    monkeypatch.setattr(inventory, "verify_completed_configuration", lambda *a, **kw: changed)
    with pytest.raises(ValueError, match="changed since inventory"):
        verify_completed_reference(config, attestation, target)


def test_legacy_dispatch_remains_available_for_new_scopes(tmp_path, monkeypatch):
    import model_creation.run_inventory as inventory
    main = tmp_path / "results/catch22/experiments/main_shared/nested_cv"
    main.mkdir(parents=True)
    _write_json(main / "run_manifest.json", {"study_identity": {"code_hash": LEGACY_CODE_HASH}})
    monkeypatch.setattr(inventory, "import_legacy_inventory", lambda *a, **kw: {"legacy": True})
    assert import_completed_inventory(tmp_path, scopes=["mortality_waveform_sensitivity"]) == [{"legacy": True}]
    monkeypatch.setattr(inventory, "verify_legacy_reference", lambda *a: {"legacy": True})
    assert verify_completed_reference({}, {"adapter": "reviewed-fcff7c7-scheduler-scope-extension-v1"}, {}) == {"legacy": True}
