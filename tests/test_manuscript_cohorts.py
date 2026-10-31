"""Manuscript population, extraction provenance, and branch assembly checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preparation.artifact_metadata import (
    ARTIFACT_SCHEMA_VERSIONS, STEP_01_COHORT_ARTIFACT, STEP_03_PREOP_ARTIFACT,
    STEP_05_MERGED_ARTIFACT, ArtifactCompatibilityError, load_artifact_metadata, metadata_path_for,
    file_sha256, write_artifact_metadata,
)
from data_preparation.cohort_scopes import build_cohort_manifest, cohort_counts, select_cohort, selected_case_hash
from data_preparation.step_01_cohort_construction import _exclude_asa_v_vi, _restrict_to_general_anesthesia
from data_preparation.step_02_catch_22 import (
    EXTRACTION_PROTOCOL_VERSION, build_extraction_fingerprint, extraction_worker_count, load_processed_pairs, _save_mode_checkpoint,
)
from data_preparation.step_04_intraop_prep import process_waveform_file
from data_preparation.step_05_data_merge import merge_and_save, write_branch_roster_manifest
from data_preparation.validate_processed_artifacts import PREOP_REQUIRED_COLUMNS, validate_prepared_artifacts


def _cohort():
    return pd.DataFrame({
        "caseid": [1, 2, 3, 4, 5], "subjectid": [10, 10, 11, 12, 13],
        "ane_type": ["General"] * 4 + ["Spinal"],
        "eligible_any_aki": [1, 0, 1, 1, 1], "aki_label": [0, np.nan, 1, 0, 0],
        "eligible_icu_admission": [1, 1, 1, 0, 1], "y_icu_admit": [0, 1, 0, np.nan, 0],
        "preop_value": [1, np.nan, 3, 4, 5],
    })


def test_balanced_sixteen_patient_holdout_requires_both_classes_in_each_partition():
    from data_preparation.step_03_preop_prep import select_patient_level_holdout_indices
    from model_creation.utils import select_patient_level_holdout_positions

    frame = pd.DataFrame({"subjectid": range(1, 17), "aki_label": [0] * 8 + [1] * 8})
    train, test = select_patient_level_holdout_indices(frame, outcome_col="aki_label", random_state=42)
    train_pos, test_pos = select_patient_level_holdout_positions(frame.aki_label, frame.subjectid, random_state=42)
    assert train.tolist() == train_pos.tolist() and test.tolist() == test_pos.tolist()
    assert frame.loc[train, "aki_label"].nunique() == frame.loc[test, "aki_label"].nunique() == 2
    assert not set(frame.loc[train, "subjectid"]) & set(frame.loc[test, "subjectid"])
    for selector in (
        lambda: select_patient_level_holdout_indices(frame.assign(aki_label=0), outcome_col="aki_label"),
        lambda: select_patient_level_holdout_positions(frame.aki_label * 0, frame.subjectid),
    ):
        with pytest.raises(ValueError, match="both classes in train and test"):
            selector()


@pytest.mark.parametrize("labels", [[0, 1, 0, 0], [0, 0, 0, 1]])
def test_persisted_holdout_rejects_single_class_training_or_test_partition(labels):
    from model_creation.utils import get_trainable_outcome_spec, require_persisted_holdout_split

    frame = pd.DataFrame({"subjectid": range(4), "aki_label": labels,
                          "split_group_any_aki": ["train", "train", "test", "test"]})
    with pytest.raises(ValueError, match="requires both classes in train and test"):
        require_persisted_holdout_split(frame, spec=get_trainable_outcome_spec("any_aki"), outcome_name="any_aki")


def test_general_anesthesia_uses_source_category_and_retains_missing_asa():
    frame = pd.DataFrame({"caseid": range(6), "ane_type": ["General"] * 3 + ["Spinal", "Sedationalgesia", None],
                          "asa": [None, 2, 5, 2, 2, 2]})
    selected = _exclude_asa_v_vi(_restrict_to_general_anesthesia(frame))
    assert selected["caseid"].tolist() == [0, 1]
    assert pd.isna(selected.iloc[0]["asa"])
    with pytest.raises(ValueError, match="ane_type"):
        _restrict_to_general_anesthesia(frame.drop(columns="ane_type"))


def test_step01_records_source_flow_without_deduplicating_operations(tmp_path, monkeypatch):
    from data_preparation import step_01_cohort_construction as construction

    clinical = pd.DataFrame({
        "caseid": [1, 2, 3, 4, 5, 6], "subjectid": [10, 10, 11, 12, 13, 14],
        "ane_type": ["General"] * 4 + ["Spinal", "Sedationalgesia"],
        "asa": [2, None, 5, 2, 2, 2], "opend": [100] * 6, "preop_cr": [1.0] * 6,
        "icu_days": [0, 1, 0, 0, 0, 0], "death_inhosp": [0] * 6, "dis": [1000] * 6,
    })
    clinical_path, lab_path, output = tmp_path / "clinical.csv", tmp_path / "labs.csv", tmp_path / "cohort.csv"
    clinical.to_csv(clinical_path, index=False)
    pd.DataFrame({"caseid": clinical.caseid, "name": ["cr"] * 6, "dt": [200] * 6,
                  "result": [1.6, 1.0, 1.0, 1.0, 1.0, 1.0]}).to_csv(lab_path, index=False)
    monkeypatch.setattr(construction, "INPUT_FILE", clinical_path)
    monkeypatch.setattr(construction, "LAB_DATA_FILE", lab_path)
    monkeypatch.setattr(construction, "COHORT_FILE", output)
    monkeypatch.setattr(construction, "COHORT_FLOW_COUNTS_FILE", tmp_path / "flow.json")
    monkeypatch.setattr(construction, "DEPARTMENTS", None)
    monkeypatch.setattr(construction, "_restrict_to_required_waveforms",
                        lambda frame: (frame.loc[frame.caseid.isin([1, 2])].copy(), []))
    construction.main()
    metadata = load_artifact_metadata(output)
    stages = metadata["outer_flow_counts"]["stages"]
    assert [stage["key"] for stage in stages] == ["source", "departments", "opend", "general_anesthesia", "asa", "waveforms"]
    assert [stage["count"] for stage in stages] == [6, 6, 6, 4, 3, 2]
    assert [stage["n_patients"] for stage in stages] == [5, 5, 5, 3, 2, 1]
    assert metadata["outer_flow_counts"]["source_file"] == str(clinical_path)
    assert pd.read_csv(output)["subjectid"].tolist() == [10, 10]


def test_scopes_match_main_outcomes_and_retain_repeat_operations_and_nans():
    frame = _cohort()
    before = frame.copy(deep=True)
    aki = select_cohort(frame, "main_shared", "any_aki")
    icu = select_cohort(frame, "main_shared", "icu_admission")
    sensitivity = select_cohort(frame, "icu_waveform_sensitivity", "icu_admission")
    assert aki["caseid"].tolist() == icu["caseid"].tolist() == [1, 3]
    assert sensitivity["caseid"].tolist() == [1, 2, 3]
    assert sensitivity["subjectid"].tolist() == [10, 10, 11]
    assert pd.isna(sensitivity.loc[1, "preop_value"])
    assert cohort_counts(sensitivity)["patients"] == 2
    assert cohort_counts(sensitivity)["outcomes"]["any_aki"] == {"operations": 2, "events": 1, "missing": 1}
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("scope,outcome", [("unknown", "any_aki"), ("icu_waveform_sensitivity", "any_aki"),
                                           ("main_shared", "mortality")])
def test_scope_rejects_unsupported_combinations(scope, outcome):
    with pytest.raises(ValueError):
        select_cohort(_cohort(), scope, outcome)


def test_roster_hash_is_order_independent_and_tracks_patient_identity():
    selected = select_cohort(_cohort(), "main_shared")
    manifest = build_cohort_manifest(_cohort(), "main_shared", "windowed")
    assert manifest["operations"] == 2
    assert manifest["patients"] == 2
    assert manifest["caseids"] == [1, 3]
    assert manifest["selected_case_hash"] == selected_case_hash(selected.iloc[::-1])
    floats = selected.astype({"caseid": float, "subjectid": float})
    assert selected_case_hash(floats) == manifest["selected_case_hash"]
    selected.loc[0, "subjectid"] = 99
    assert selected_case_hash(selected) != manifest["selected_case_hash"]
    with pytest.raises(ValueError, match="duplicate caseid"):
        selected_case_hash(pd.concat([selected, selected.iloc[[0]]]))
    selected.loc[0, "subjectid"] = np.nan
    with pytest.raises(ValueError, match="nonmissing"):
        select_cohort(selected, "main_shared")


def test_extraction_resume_requires_matching_method_and_operation_roster(tmp_path, monkeypatch):
    from data_preparation import step_02_catch_22 as extraction

    cohort = pd.DataFrame({"caseid": [1], "opstart": [0], "opend": [100]})
    fingerprint = build_extraction_fingerprint(cohort, "windowed")
    output, errors = tmp_path / "features.csv", tmp_path / "errors.csv"
    pd.DataFrame({"caseid": [1], "waveform": ["SNUADC/PLETH"], "DN_Mean": [1]}).to_csv(output, index=False)
    with pytest.raises(ArtifactCompatibilityError, match="no extraction fingerprint"):
        load_processed_pairs(output, errors, fingerprint)
    metadata_path_for(output).write_text(json.dumps({"extraction_fingerprint": fingerprint}))
    assert load_processed_pairs(output, errors, fingerprint) == {(1, "SNUADC/PLETH")}
    changed_cohort = cohort.assign(opend=101)
    with pytest.raises(ArtifactCompatibilityError, match="mismatch"):
        load_processed_pairs(output, errors, build_extraction_fingerprint(changed_cohort, "windowed"))
    monkeypatch.setattr(extraction, "WIN_SEC", extraction.WIN_SEC + 1)
    with pytest.raises(ArtifactCompatibilityError, match="mismatch"):
        load_processed_pairs(output, errors, build_extraction_fingerprint(cohort, "windowed"))
    pd.DataFrame({"caseid": [1, 1], "waveform": ["SNUADC/PLETH"] * 2}).to_csv(output, index=False)
    with pytest.raises(ArtifactCompatibilityError, match="Duplicate"):
        load_processed_pairs(output, errors, fingerprint)


def test_checkpoint_preserves_nan_features_and_drains_buffers(tmp_path):
    output, errors = tmp_path / "features.csv", tmp_path / "errors.csv"
    conf = {"file": output, "error_file": errors, "features": [{"caseid": 1, "waveform": "SNUADC/PLETH", "feature": np.nan}],
            "errors": [], "processed_pairs": {(1, "SNUADC/PLETH")}}
    _save_mode_checkpoint(conf, {"test": "fingerprint"})
    assert pd.isna(pd.read_csv(output).loc[0, "feature"])
    assert conf["features"] == []
    _save_mode_checkpoint(conf, {"test": "fingerprint"})
    assert len(pd.read_csv(output)) == 1


def test_interrupted_checkpoint_write_leaves_previous_snapshot_reusable(tmp_path, monkeypatch):
    output, errors = tmp_path / "features.csv", tmp_path / "errors.csv"
    conf = {"file": output, "error_file": errors, "features": [{"caseid": 1, "waveform": "SNUADC/PLETH", "feature": 2.0}],
            "errors": [], "processed_pairs": {(1, "SNUADC/PLETH")}}
    fingerprint = {"test": "fingerprint"}
    _save_mode_checkpoint(conf, fingerprint)
    original = output.read_bytes()
    conf["features"] = [{"caseid": 2, "waveform": "SNUADC/PLETH", "feature": 3.0}]
    conf["processed_pairs"].add((2, "SNUADC/PLETH"))

    def interrupted_write(_frame, path, **_kwargs):
        Path(path).write_text("caseid,waveform,feature\n2,SNUADC/PLETH,")
        raise OSError("simulated interruption")

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted_write)
    with pytest.raises(OSError, match="interruption"):
        _save_mode_checkpoint(conf, fingerprint)
    assert output.read_bytes() == original
    assert load_processed_pairs(output, errors, fingerprint) == {(1, "SNUADC/PLETH")}


def test_checkpoint_feature_values_do_not_depend_on_batching_or_resumption(tmp_path):
    records = [
        {"caseid": caseid, "waveform": "SNUADC/PLETH", "feature_mean": np.float32(value),
         "feature_std": np.float32(value / 7), "float64_feature": float(value) / 11}
        for caseid, value in enumerate((1.23456789, 0.0000123456789, 123456.789, -0.07123456), 1)
    ]
    fingerprint = {"test": "serializer"}

    def new_configuration(name):
        return {"file": tmp_path / f"{name}.csv", "error_file": tmp_path / f"{name}_errors.csv",
                "features": [], "errors": [], "processed_pairs": set()}

    single = new_configuration("single")
    single["features"] = records.copy()
    single["processed_pairs"] = {(row["caseid"], row["waveform"]) for row in records}
    _save_mode_checkpoint(single, fingerprint)
    batched = new_configuration("batched")
    for record in records:
        batched["features"] = [record]
        batched["processed_pairs"].add((record["caseid"], record["waveform"]))
        _save_mode_checkpoint(batched, fingerprint)
    resumed = new_configuration("resumed")
    resumed["features"] = records[:2]
    resumed["processed_pairs"] = {(row["caseid"], row["waveform"]) for row in records[:2]}
    _save_mode_checkpoint(resumed, fingerprint)
    resumed["processed_pairs"] = load_processed_pairs(resumed["file"], resumed["error_file"], fingerprint)
    resumed["features"] = records[2:]
    resumed["processed_pairs"].update((row["caseid"], row["waveform"]) for row in records[2:])
    _save_mode_checkpoint(resumed, fingerprint)

    assert single["file"].read_bytes() == batched["file"].read_bytes() == resumed["file"].read_bytes()
    restored = pd.read_csv(single["file"], float_precision="round_trip")
    expected = pd.DataFrame(records)
    for column in ("feature_mean", "feature_std", "float64_feature"):
        np.testing.assert_array_equal(restored[column].to_numpy(), expected[column].to_numpy(dtype=np.float64))
    assert restored["caseid"].tolist() == [1, 2, 3, 4]


def _write_long_with_provenance(path, frame):
    frame.to_csv(path, index=False)
    metadata_path_for(path).write_text(json.dumps({"extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION,
                                                 "extraction_fingerprint": {"test": "fingerprint",
                                                    "extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION}}))


def test_wide_assembly_rejects_duplicate_pairs(tmp_path):
    source = tmp_path / "long.csv"
    _write_long_with_provenance(source, pd.DataFrame({"caseid": [1, 1], "waveform": ["SNUADC/PLETH"] * 2,
                                                    "feature": [1, 3]}))
    with pytest.raises(ValueError, match="Duplicate case/channel"):
        process_waveform_file(source, tmp_path / "wide.csv", "Full Features")


def test_wide_assembly_and_merge_preserve_branch_population_and_missing_channels(tmp_path):
    source, wide, merged = tmp_path / "long.csv", tmp_path / "wide.csv", tmp_path / "merged.csv"
    _write_long_with_provenance(source, pd.DataFrame({"caseid": [1, 2], "waveform": ["SNUADC/PLETH"] * 2,
                                                    "feature": [1, np.nan]}))
    process_waveform_file(source, wide, "Windowed Features")
    features = pd.read_csv(wide)
    assert features["caseid"].tolist() == [1, 2]
    assert "Primus_AWP_feature" in features
    assert features["Primus_AWP_feature"].isna().all()
    assert json.loads(metadata_path_for(wide).read_text())["extraction_diagnostics"]["operations_with_partial_channels"] == 2
    flow = {"source_file": "source.csv", "stages": [{"key": "source", "title": "All operations", "count": 10, "n_patients": 9}]}
    trim = {"operations_before": 10, "operations_after": 5, "patients_after": 4}
    source_scopes = {scope: build_cohort_manifest(_cohort(), scope)
                     for scope in ("main_shared", "icu_waveform_sensitivity")}
    merge_and_save(_cohort(), wide, merged, "Windowed Features", False, require_intraop_metadata=True,
                   preop_metadata={"outer_flow_counts": flow, "diagnostic_trim": trim, "cohort_scopes": source_scopes})
    metadata = load_artifact_metadata(merged)
    assert metadata["branch"] == "windowed"
    assert metadata["cohort_scopes"]["main_shared"]["caseids"] == [1]
    assert metadata["cohort_scopes"]["icu_waveform_sensitivity"]["caseids"] == [1, 2]
    assert metadata["cohort_scopes"]["icu_waveform_sensitivity"]["patients"] == 1
    assert metadata["outer_flow_counts"] == flow
    assert metadata["diagnostic_trim"] == trim
    assert metadata["source_cohort_scopes"] == source_scopes
    pd.DataFrame({"caseid": [1], "untrusted_feature": [0]}).to_csv(wide, index=False)
    with pytest.raises(ArtifactCompatibilityError, match="hash mismatch"):
        merge_and_save(_cohort(), wide, merged, "Windowed Features", False, require_intraop_metadata=True)


def test_step04_rejects_unidentified_historical_features(tmp_path):
    source = tmp_path / "long.csv"
    pd.DataFrame({"caseid": [1], "waveform": ["SNUADC/PLETH"], "feature": [1]}).to_csv(source, index=False)
    with pytest.raises(ArtifactCompatibilityError, match="lacks extraction provenance"):
        process_waveform_file(source, tmp_path / "wide.csv", "Full Features")


@pytest.mark.parametrize("previous_protocol", [1, 2])
def test_step04_rejects_previous_extraction_protocols(tmp_path, previous_protocol):
    source = tmp_path / "long.csv"
    _write_long_with_provenance(source, pd.DataFrame({"caseid": [1], "waveform": ["SNUADC/PLETH"], "feature": [1]}))
    payload = json.loads(metadata_path_for(source).read_text())
    payload["extraction_protocol_version"] = previous_protocol
    metadata_path_for(source).write_text(json.dumps(payload))
    with pytest.raises(ArtifactCompatibilityError, match="supported extraction fingerprint"):
        process_waveform_file(source, tmp_path / "wide.csv", "Full Features")


@pytest.mark.parametrize("previous_protocol", [1, 2])
def test_merge_rejects_old_assembled_features_before_writing(tmp_path, previous_protocol):
    wide, output = tmp_path / "wide.csv", tmp_path / "merged.csv"
    pd.DataFrame({"caseid": [1], "Primus_CO2_feature": [1.]}).to_csv(wide, index=False)
    metadata_path_for(wide).write_text(json.dumps({"assembly_protocol_version": 1,
        "data_sha256": file_sha256(wide),
        "extraction_fingerprint": {"extraction_protocol_version": previous_protocol}}))
    with pytest.raises(ArtifactCompatibilityError, match="Stale waveform extraction protocol"):
        merge_and_save(_cohort(), wide, output, "Full Features", False, require_intraop_metadata=True)
    assert not output.exists()


@pytest.mark.parametrize("branch", ["non_windowed", "windowed"])
def test_model_loader_rejects_pre_antialias_merged_schema(tmp_path, monkeypatch, branch):
    from model_creation import utils
    path = tmp_path / "old_merged.csv"
    pd.DataFrame({column: [1] for column in utils.MERGED_REQUIRED_COLUMNS}).to_csv(path, index=False)
    write_artifact_metadata(path, artifact_role=STEP_05_MERGED_ARTIFACT,
        available_columns=utils.MERGED_REQUIRED_COLUMNS, extra_metadata={"schema_version": 4})
    monkeypatch.setattr(utils, "FULL_FEATURES_FILE", path)
    monkeypatch.setattr(utils, "WINDOWED_FEATURES_FILE", path)
    with pytest.raises(ArtifactCompatibilityError, match="expected schema_version 5 but found 4"):
        utils.load_data(branch)


def test_branch_rosters_record_intersections_without_changing_branch_populations(tmp_path):
    full, windowed, output = tmp_path / "full.csv", tmp_path / "windowed.csv", tmp_path / "cohort_rosters.json"
    for path, frame in ((full, _cohort().iloc[:3]), (windowed, _cohort().iloc[[1, 2]])):
        frame.to_csv(path, index=False)
        write_artifact_metadata(path, artifact_role=STEP_05_MERGED_ARTIFACT, available_columns=frame.columns)
    paths = {"non_windowed": full, "windowed": windowed}
    write_branch_roster_manifest(paths, output)
    payload = json.loads(output.read_text())
    assert payload["main_shared"]["branches"]["non_windowed"]["caseids"] == [1, 3]
    assert payload["main_shared"]["branches"]["windowed"]["caseids"] == [3]
    assert payload["main_shared"]["intersection"]["caseids"] == [3]
    assert payload["icu_waveform_sensitivity"]["intersection"]["caseids"] == [2, 3]
    corrupted = pd.read_csv(windowed)
    corrupted.loc[corrupted["caseid"].eq(3), "subjectid"] = 999
    corrupted.to_csv(windowed, index=False)
    with pytest.raises(ValueError, match="Patient identity differs"):
        write_branch_roster_manifest(paths, output)


def test_prepared_reuse_validates_sources_content_and_scope_rosters(tmp_path, monkeypatch):
    clinical, labs = tmp_path / "clinical.csv", tmp_path / "labs.csv"
    clinical.write_text("clinical source snapshot\n")
    labs.write_text("lab source snapshot\n")
    hashes = {"clinical_data_sha256": file_sha256(clinical), "lab_data_sha256": file_sha256(labs)}
    frame = _cohort()
    frame["opstart"], frame["opend"] = 0, 100
    for column in PREOP_REQUIRED_COLUMNS:
        if column not in frame:
            frame[column] = 0
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    write_artifact_metadata(cohort_path, artifact_role=STEP_01_COHORT_ARTIFACT, available_columns=frame.columns)
    preop, full, windowed = tmp_path / "preop.csv", tmp_path / "full.csv", tmp_path / "windowed.csv"
    frame.to_csv(preop, index=False)
    write_artifact_metadata(preop, artifact_role=STEP_03_PREOP_ARTIFACT, available_columns=frame.columns,
                            extra_metadata={"source_hashes": hashes, "data_sha256": file_sha256(preop),
                                            "upstream_schema_version": ARTIFACT_SCHEMA_VERSIONS[STEP_01_COHORT_ARTIFACT]})
    paths = {"non_windowed": full, "windowed": windowed}
    for branch, path in paths.items():
        frame.to_csv(path, index=False)
        write_artifact_metadata(path, artifact_role=STEP_05_MERGED_ARTIFACT, available_columns=frame.columns,
                                extra_metadata={"source_hashes": hashes, "data_sha256": file_sha256(path),
                                                "upstream_schema_version": ARTIFACT_SCHEMA_VERSIONS[STEP_03_PREOP_ARTIFACT],
                                                "branch": branch, "extraction_fingerprint": build_extraction_fingerprint(
                                                    frame, "full" if branch == "non_windowed" else "windowed"),
                                                "cohort_scopes": {scope: build_cohort_manifest(frame, scope, branch)
                                                                  for scope in ("main_shared", "icu_waveform_sensitivity")}})
    sources = {"clinical_data_sha256": clinical, "lab_data_sha256": labs}
    validate_prepared_artifacts(preop, paths, cohort_path=cohort_path, source_paths=sources)
    # Saved-artifact replay remains valid when research sources are unavailable.
    validate_prepared_artifacts(preop, paths, cohort_path=cohort_path,
                                source_paths={"clinical_data_sha256": tmp_path / "absent.csv"})
    # Reuse cannot bypass the Step 02 contract merely because merged CSVs exist.
    windowed_metadata = load_artifact_metadata(windowed)
    for field, incompatible_value in (
        ("extraction_protocol_version", 999), ("mode", "full"),
        ("cohort_roster_sha256", "stale"), ("code_sha256", {"old.py": "stale"}),
        ("pycatch22_version", "old"), ("signal_specs", {}),
        ("window_seconds", 30), ("slide_seconds", 10), ("full_target_sr", 1),
    ):
        corrupted_metadata = json.loads(json.dumps(windowed_metadata))
        corrupted_metadata["extraction_fingerprint"][field] = incompatible_value
        metadata_path_for(windowed).write_text(json.dumps(corrupted_metadata))
        with pytest.raises(ArtifactCompatibilityError, match="extraction fingerprint mismatch"):
            validate_prepared_artifacts(preop, paths, cohort_path=cohort_path)
    metadata_path_for(windowed).write_text(json.dumps(windowed_metadata))
    from data_preparation import step_02_catch_22 as extraction
    with monkeypatch.context() as settings:
        settings.setattr(extraction, "WIN_SEC", 30)
        with pytest.raises(ArtifactCompatibilityError, match="extraction fingerprint mismatch"):
            validate_prepared_artifacts(preop, paths, cohort_path=cohort_path)
    changed_cohort = frame.copy()
    changed_cohort.loc[0, "opend"] += 1
    changed_cohort.to_csv(cohort_path, index=False)
    with pytest.raises(ArtifactCompatibilityError, match="extraction fingerprint mismatch"):
        validate_prepared_artifacts(preop, paths, cohort_path=cohort_path)
    frame.to_csv(cohort_path, index=False)
    for column, value in (("subjectid", 999), ("aki_label", 1), ("eligible_any_aki", 0)):
        changed_cohort = frame.copy()
        changed_cohort.loc[0, column] = value
        changed_cohort.to_csv(cohort_path, index=False)
        with pytest.raises(ArtifactCompatibilityError, match="Patient/outcome/eligibility values disagree"):
            validate_prepared_artifacts(preop, paths, cohort_path=cohort_path)
    frame.to_csv(cohort_path, index=False)
    # Check the merged branch independently of the Step 03 artifact as well.
    changed_branch = frame.copy()
    changed_branch.loc[0, "subjectid"] = 999
    original_metadata = load_artifact_metadata(full)
    changed_branch.to_csv(full, index=False)
    changed_metadata = json.loads(json.dumps(original_metadata))
    changed_metadata["data_sha256"] = file_sha256(full)
    changed_metadata["cohort_scopes"] = {
        scope: build_cohort_manifest(changed_branch, scope, "non_windowed")
        for scope in ("main_shared", "icu_waveform_sensitivity")
    }
    metadata_path_for(full).write_text(json.dumps(changed_metadata))
    with pytest.raises(ArtifactCompatibilityError, match="Patient/outcome/eligibility values disagree"):
        validate_prepared_artifacts(None, paths, cohort_path=cohort_path)
    frame.to_csv(full, index=False)
    metadata_path_for(full).write_text(json.dumps(original_metadata))
    labs.write_text("changed lab snapshot\n")
    with pytest.raises(ArtifactCompatibilityError, match="Raw source changed"):
        validate_prepared_artifacts(preop, paths, cohort_path=cohort_path, source_paths=sources)
    labs.write_text("lab source snapshot\n")
    metadata = load_artifact_metadata(full)
    metadata["source_hashes"]["lab_data_sha256"] = "different"
    metadata_path_for(full).write_text(json.dumps(metadata))
    with pytest.raises(ArtifactCompatibilityError, match="disagree on raw source"):
        validate_prepared_artifacts(preop, paths, cohort_path=cohort_path)
    metadata["source_hashes"] = hashes
    metadata["cohort_scopes"]["main_shared"]["caseids"] = [3]
    metadata_path_for(full).write_text(json.dumps(metadata))
    with pytest.raises(ArtifactCompatibilityError, match="roster/count metadata mismatch"):
        validate_prepared_artifacts(preop, paths, cohort_path=cohort_path)
    metadata["cohort_scopes"]["main_shared"] = build_cohort_manifest(frame, "main_shared", "non_windowed")
    metadata_path_for(full).write_text(json.dumps(metadata))
    changed_frame = pd.read_csv(full)
    changed_frame.loc[0, "preop_value"] = 999
    changed_frame.to_csv(full, index=False)
    with pytest.raises(ArtifactCompatibilityError, match="CSV content hash mismatch"):
        validate_prepared_artifacts(preop, paths, cohort_path=cohort_path)


def test_step05_legacy_imputation_cli_preserves_missing_outcomes_and_features(tmp_path, monkeypatch, capsys):
    from data_preparation import step_05_data_merge as merge

    frame = _cohort()
    for column in PREOP_REQUIRED_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    preop_path = tmp_path / "preop.csv"
    frame.to_csv(preop_path, index=False)
    write_artifact_metadata(preop_path, artifact_role=STEP_03_PREOP_ARTIFACT, available_columns=frame.columns)
    monkeypatch.setattr(merge, "PREOP_PROCESSED_FILE", preop_path)
    for suffix, input_key, output_key in (
        ("full", "INTRAOP_WIDE_FILE", "WIDE_FEATURES_FILE"),
        ("windowed", "INTRAOP_WIDE_WINDOWED_FILE", "WIDE_FEATURES_WINDOWED_FILE"),
    ):
        wide, output = tmp_path / f"{suffix}_wide.csv", tmp_path / f"{suffix}_merged.csv"
        pd.DataFrame({"caseid": frame.caseid, "wave_feature": [1, np.nan, 3, 4, 5]}).to_csv(wide, index=False)
        metadata_path_for(wide).write_text(json.dumps({"assembly_protocol_version": 1, "data_sha256": file_sha256(wide),
            "extraction_fingerprint": {"extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION}}))
        monkeypatch.setattr(merge, input_key, wide)
        monkeypatch.setattr(merge, output_key, output)
    monkeypatch.setattr(merge.sys, "argv", ["step_05_data_merge", "--impute-missing"])
    merge.main()
    for path in (merge.WIDE_FEATURES_FILE, merge.WIDE_FEATURES_WINDOWED_FILE):
        result = pd.read_csv(path)
        assert pd.isna(result.loc[1, "aki_label"])
        assert pd.isna(result.loc[3, "y_icu_admit"])
        assert pd.isna(result.loc[1, "wave_feature"])
        assert pd.isna(result.loc[1, "preop_value"])
        assert pd.isna(result.loc[1, "split_group_any_aki"])
        assert not result.select_dtypes(include="number").eq(-99).any().any()
    assert "CLI compatibility only" in capsys.readouterr().out


@pytest.mark.parametrize("requested,tasks,expected", [(None, 100, 2), (16, 100, 2), (1, 100, 1), (16, 1, 1)])
def test_extraction_workers_respect_two_cpu_quota_despite_64_cpu_affinity(monkeypatch, requested, tasks, expected):
    import runtime_resources

    original_read_text = Path.read_text

    def quota_read_text(path, *args, **kwargs):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            return "200000 100000\n"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(runtime_resources.os, "sched_getaffinity", lambda _pid: set(range(64)))
    monkeypatch.setattr(Path, "read_text", quota_read_text)
    if requested is None:
        monkeypatch.delenv("AKI_EXTRACTION_WORKERS", raising=False)
    else:
        monkeypatch.setenv("AKI_EXTRACTION_WORKERS", str(requested))
    assert extraction_worker_count(tasks) == expected
