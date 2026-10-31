from pathlib import Path

import pytest

import run_catch22


@pytest.mark.parametrize("validation_result", [0, 1])
def test_preparation_validates_completed_outputs_before_training(tmp_path, monkeypatch, validation_result):
    calls = []

    def fake_run_module(module, args, *, env, log_handle):
        calls.append(module)
        assert env["GENERATE_FULL_FEATURES"] == env["GENERATE_WINDOWED_FEATURES"] == "True"
        return validation_result if module == "data_preparation.validate_processed_artifacts" else 0

    monkeypatch.setattr(run_catch22, "run_module", fake_run_module)
    result = run_catch22._ensure_data_prepared(
        "auto", env={}, processed_dir=tmp_path / "processed", log_handle=None
    )
    assert result == validation_result
    assert calls == [
        "data_preparation.step_01_cohort_construction", "data_preparation.step_02_catch_22",
        "data_preparation.step_03_preop_prep", "data_preparation.step_04_intraop_prep",
        "data_preparation.step_05_data_merge", "data_preparation.validate_processed_artifacts",
    ]


def test_preparation_uses_directory_roles_for_dotted_roots_and_extensionless_logs(tmp_path):
    paths = {"processed_dir": tmp_path / "study.v1/processed.v1",
             "results_dir": tmp_path / "study.v1/results.v1",
             "log_file": tmp_path / "logs/run-log"}
    run_catch22._ensure_paths(paths)
    assert paths["processed_dir"].is_dir() and paths["results_dir"].is_dir()
    assert paths["log_file"].parent.is_dir()
    assert not paths["log_file"].exists()


def _write_prepared_fixture(processed_dir):
    import pandas as pd
    from data_preparation.artifact_metadata import STEP_05_MERGED_ARTIFACT, write_artifact_metadata
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"caseid": [1, 2], "subjectid": [101, 102], "aki_label": [0, 1],
                          "eligible_any_aki": [1, 1], "y_icu_admit": [1, 0], "eligible_icu_admission": [1, 1],
                          "eligible_mortality": [1, 1], "y_inhosp_mortality": [0, 1]})
    for suffix in ("", "_windowed"):
        path = processed_dir / f"aki_features_master_wide{suffix}.csv"
        frame.to_csv(path, index=False)
        write_artifact_metadata(path, artifact_role=STEP_05_MERGED_ARTIFACT, available_columns=frame.columns)



def _mock_experiment_queue(monkeypatch, calls, *, exit_code=0, verify=True):
    """Replace process execution, retaining CLI construction and status callbacks."""
    from model_creation import experiment_queue, split_plan
    completed = set()
    def prepare(processed, configs, *a, **k):
        return {"status": "validated", "fold_assignment_hashes": {
            "/".join(c[field] for field in ("cohort_scope", "outcome", "branch")): "test-frozen-folds"
            for c in configs}}
    monkeypatch.setattr(split_plan, "prepare_split_plan", prepare)
    def queue(jobs, *, on_finished, **kwargs):
        records = []
        for job in jobs:
            calls.append(("model_creation.step_07_train_evaluate", list(job.command[3:])))
            record = {"key": job.key, "exit_code": exit_code,
                      "status": "complete" if exit_code == 0 else "failed"}
            if verify and exit_code == 0:
                completed.add(job.key)
            on_finished(record)
            records.append(record)
        return records
    monkeypatch.setattr(experiment_queue, "run_configurations", queue)
    if verify:
        from model_creation.run_inventory import configuration_key
        monkeypatch.setattr(experiment_queue, "_completed_reference",
                            lambda args, config, *a, **k: {"status": "verified_completed"}
                            if configuration_key(config) in completed else None)


def test_experiments_fail_fast_when_generated_paths_are_off_media(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "data" / "processed"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results" / "catch22" / "experiments"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "results" / "catch22" / "paper"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "experiment.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "enforce")
    from functools import partial
    monkeypatch.setattr(run_catch22, "enforce_storage_policy",
                        partial(run_catch22.enforce_storage_policy, allowed_root=tmp_path / "allowed-volume"))
    monkeypatch.setattr(run_catch22, "refresh_repo_convenience_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_catch22,
        "run_module",
        lambda *args, **kwargs: pytest.fail("run_module should not be called when storage policy fails"),
    )

    result = run_catch22.main(["experiments", "--prep", "skip", "--only-xgboost", "--no-resume"])

    captured = capsys.readouterr()
    assert result == 1
    assert f"Generated artifact paths must live under {tmp_path / 'allowed-volume'}" in captured.err
    assert "Starting Experiments" not in captured.out


def test_experiments_skip_reporting_on_validation_failures(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "data" / "processed"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results" / "catch22" / "experiments"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "results" / "catch22" / "paper"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "experiment.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr(run_catch22, "refresh_repo_convenience_paths", lambda *args, **kwargs: None)

    calls: list[tuple[str, list[str]]] = []

    def fake_run_module(module: str, args: list[str], *, env, log_handle) -> int:
        calls.append((module, list(args)))
        if module == "data_preparation.validate_processed_artifacts":
            _write_prepared_fixture(env["PROCESSED_DIR"])
            return 0
        if module == "model_creation.step_07_train_evaluate":
            return 1
        pytest.fail(f"Unexpected module call: {module}")

    monkeypatch.setattr(run_catch22, "run_module", fake_run_module)
    _mock_experiment_queue(monkeypatch, calls, exit_code=1)

    result = run_catch22.main(["experiments", "--prep", "skip", "--only-xgboost", "--no-resume"])

    captured = capsys.readouterr()
    assert result == 1
    assert any(module == "data_preparation.validate_processed_artifacts" for module, _ in calls)
    assert any(module == "model_creation.step_07_train_evaluate" for module, _ in calls)
    assert "skipping metrics/report generation" in captured.out
    import json
    saved = json.loads((tmp_path / "results/catch22/experiments/main_shared/nested_cv/run_manifest.json").read_text())
    assert saved["status"] == "failed"
    assert saved["execution_history"] and all(row["exit_code"] == 1 for row in saved["execution_history"])
    assert "Running evaluation (results -> results directory)" not in captured.out


def test_smoke_uses_isolated_root_and_selected_models(tmp_path: Path, monkeypatch):
    smoke_root = tmp_path / "smoke"
    monkeypatch.setenv("SMOKE_ROOT", str(smoke_root))
    monkeypatch.setenv("LOG_FILE", str(smoke_root / "smoke_test.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr(run_catch22, "refresh_repo_convenience_paths", lambda *args, **kwargs: None)

    calls: list[tuple[str, list[str], str, str]] = []

    def fake_run_module(module: str, args: list[str], *, env, log_handle) -> int:
        calls.append((module, list(args), env["PROCESSED_DIR"], env["RESULTS_DIR"]))
        if module == "data_preparation.step_05_data_merge":
            _write_prepared_fixture(env["PROCESSED_DIR"])
        return 0

    monkeypatch.setattr(run_catch22, "run_module", fake_run_module)

    result = run_catch22.main(
        [
            "smoke",
            "--outcomes",
            "any_aki",
            "--model-types",
            "xgboost,ebm",
            "--case-limit",
            "6",
            "--hpo-trials",
            "1",
        ]
    )

    assert result == 0
    trim_calls = [entry for entry in calls if entry[0] == "data_preparation.smoke_trim_cohort"]
    assert len(trim_calls) == 1
    trim_module, trim_args, processed_dir, results_dir = trim_calls[0]
    assert trim_module == "data_preparation.smoke_trim_cohort"
    assert str(smoke_root / "data" / "processed" / "aki_pleth_ecg_co2_awp.csv") in trim_args
    assert processed_dir == str(smoke_root / "data" / "processed")
    scoped_results = smoke_root / "results" / "catch22" / "experiments" / "main_shared" / "holdout"
    assert results_dir == str(scoped_results)

    validation_calls = [entry for entry in calls if entry[0] == "model_creation.step_07_train_evaluate"]
    assert len(validation_calls) == 6
    assert {call[1][call[1].index("--model_type") + 1] for call in validation_calls} == {"xgboost", "ebm"}
    assert all(call[2] == str(smoke_root / "data" / "processed") for call in validation_calls)
    assert all(call[3] == str(scoped_results) for call in validation_calls)


def test_descriptive_fails_clearly_on_missing_inputs(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "data" / "processed"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results" / "catch22" / "experiments"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "results" / "catch22" / "paper"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "descriptive.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr(run_catch22, "refresh_repo_convenience_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_catch22,
        "run_module",
        lambda *args, **kwargs: pytest.fail("run_module should not be called when required inputs are missing"),
    )

    result = run_catch22.main(["descriptive"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Required file not found" in captured.out


@pytest.mark.parametrize("preset,count", [("core", 36), ("full", 180)])
def test_dry_run_resolves_grid_without_writing(preset, count, tmp_path, monkeypatch, capsys):
    import json
    root = tmp_path / "untouched"
    for variable, suffix in (("AKI_ARTIFACT_ROOT", "artifacts"), ("PROCESSED_DIR", "processed"),
                             ("RESULTS_DIR", "results"), ("PAPER_DIR", "paper"), ("LOG_FILE", "log.txt")):
        monkeypatch.setenv(variable, str(root / suffix))
    monkeypatch.setattr(run_catch22, "run_module", lambda *a, **k: pytest.fail("dry run executed a module"))
    assert run_catch22.main(["experiments", "--preset", preset, "--dry-run"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert len(manifest["configurations"]) == count
    assert (manifest["outer_folds"], manifest["inner_folds"], manifest["calibration_folds"], manifest["repeats"]) == (5, 3, 5, 1)
    assert {c["model_type"]: c["n_trials"] for c in manifest["configurations"]} == {
        "logistic_regression": 20, "xgboost": 50, "ebm": 50,
    }
    assert manifest["reporting"] == {"bootstrap_replicates": 5000, "bootstrap_seed": 0,
                                    "bootstrap_stratified": False}
    assert manifest["results_dir"].endswith("main_shared/nested_cv")
    assert not root.exists()


def test_core_allowlists_and_sensitivity_dry_run(tmp_path, monkeypatch, capsys):
    import json
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "paper"))
    assert run_catch22.main(["experiments", "--preset", "core", "--cohort-scope", "icu_waveform_sensitivity",
                            "--model-types", "logistic_regression", "--branches", "windowed", "--dry-run"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert len(manifest["configurations"]) == 3
    assert {c["outcome"] for c in manifest["configurations"]} == {"icu_admission"}
    assert [c["feature_set"] for c in manifest["configurations"]] == [
        "preop_only", "all_waveforms", "preop_and_all_waveforms",
    ]


@pytest.mark.parametrize("arguments,message", [
    (["--outcomes", "any_aki", "--cohort-scope", "icu_waveform_sensitivity"],
     "icu_waveform_sensitivity supports only icu_admission"),
    (["--feature-sets", "unknown"], "Invalid feature sets"),
    (["--outer-folds", "2"], "Definitive validation requires"),
    (["--inner-folds", "5"], "Definitive validation requires"),
    (["--calibration-folds", "3"], "Definitive validation requires"),
], ids=["scope-outcome", "unknown-feature-set", "outer-folds", "hpo-folds", "calibration-folds"])
def test_invalid_study_configurations_fail_before_work(arguments, message, tmp_path, monkeypatch, capsys):
    root = tmp_path / "untouched"
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(root))
    monkeypatch.setattr(run_catch22, "run_module", lambda *a, **k: pytest.fail("Invalid run executed a module"))
    assert run_catch22.main(["experiments", "--dry-run", *arguments]) == 1
    assert message in capsys.readouterr().err
    assert not root.exists()


def test_staged_families_extend_manifest_without_duplicate_configs(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "paper"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "run.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    calls = []
    def fake_run(module, args, **kwargs):
        calls.append((module, args))
        if module == "data_preparation.validate_processed_artifacts":
            _write_prepared_fixture(kwargs["env"]["PROCESSED_DIR"])
        return 0
    monkeypatch.setattr(run_catch22, "run_module", fake_run)
    _mock_experiment_queue(monkeypatch, calls)
    common = ["experiments", "--preset", "core", "--prep", "skip", "--outcomes", "any_aki",
              "--branches", "non_windowed", "--skip-report"]
    for family in ("logistic_regression", "xgboost", "ebm", "logistic_regression"):
        assert run_catch22.main(common + ["--model-types", family]) == 0
    manifest = json.loads((tmp_path / "results/main_shared/nested_cv/run_manifest.json").read_text())
    assert len(manifest["configurations"]) == 9
    assert manifest["status"] == "models_complete"  # bundle validation is explicitly mocked above
    assert len(manifest["execution_history"]) == 9  # repeated completed family launches zero jobs
    assert not any(module.startswith("reporting") for module, _ in calls)
    assert all("--export_ebm_explanations" not in args for module, args in calls if module.endswith("step_07_train_evaluate"))
    assert all("--cohort-scope" in args for module, args in calls if module.endswith("step_07_train_evaluate"))
    for module, args in calls:
        if module.endswith("step_07_train_evaluate"):
            assert args[args.index("--inner-folds") + 1] == "3"
            assert args[args.index("--calibration-folds") + 1] == "5"
    assert run_catch22.main(common + ["--model-types", "logistic_regression", "--n-trials", "5"]) == 1


def test_report_only_never_prepares_or_fits_models_and_records_failure(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "paper"))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "report.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    manifest_path = tmp_path / "results/main_shared/nested_cv/run_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"configurations": [{"model_type": "logistic_regression"}], "status": "models_complete"}))
    calls = []
    def fake_run(module, args, **kwargs):
        calls.append(module)
        assert module == "results_recreation.metrics_summary"
        return 1
    monkeypatch.setattr(run_catch22, "run_module", fake_run)
    assert run_catch22.main(["experiments", "--report-only", "--bootstrap", "37",
                            "--bootstrap-seed", "19", "--bootstrap-stratified"]) == 1
    assert calls == ["results_recreation.metrics_summary"]
    assert json.loads(manifest_path.read_text())["status"] == "reporting_failed"
    assert json.loads(manifest_path.read_text())["reporting"] == {
        "bootstrap_replicates": 37, "bootstrap_seed": 19, "bootstrap_stratified": True,
    }


@pytest.mark.parametrize("stratified", [False, True])
def test_reporting_forwards_bootstrap_policy_without_training(tmp_path, monkeypatch, stratified):
    calls = []
    monkeypatch.setattr(run_catch22, "run_module", lambda module, args, **kwargs: calls.append((module, args)) or 0)
    assert run_catch22._run_reporting(tmp_path, "threads", env={}, log_handle=None,
        bootstrap=53, bootstrap_seed=17, bootstrap_stratified=stratified) == 0
    assert [module for module, args in calls] == ["results_recreation.metrics_summary", "reporting.make_report"]
    args = calls[0][1]
    assert args[args.index("--bootstrap") + 1] == "53"
    assert args[args.index("--bootstrap-seed") + 1] == "17"
    assert ("--bootstrap-stratified" in args) == stratified
    assert ("--no-bootstrap-stratified" in args) != stratified


def test_reporting_policy_dry_run_and_validation_are_side_effect_free(tmp_path, monkeypatch, capsys):
    import json
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(tmp_path / "untouched"))
    assert run_catch22.main(["experiments", "--dry-run", "--bootstrap", "73", "--bootstrap-seed", "9",
                            "--bootstrap-stratified"]) == 0
    assert json.loads(capsys.readouterr().out)["reporting"] == {
        "bootstrap_replicates": 73, "bootstrap_seed": 9, "bootstrap_stratified": True,
    }
    assert run_catch22.main(["experiments", "--dry-run", "--bootstrap", "-1"]) == 1
    assert run_catch22.main(["experiments", "--dry-run", "--bootstrap-seed", "-1"]) == 1
    assert not (tmp_path / "untouched").exists()


def test_launcher_discards_inherited_per_file_output_overrides(tmp_path, monkeypatch):
    for name in ("DATA_DIR", "PROCESSED_DIR", "RESULTS_DIR", "PAPER_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COHORT_FILE", str(tmp_path / "historical.csv"))
    monkeypatch.setenv("CATCH_22_FILE", str(tmp_path / "historical-features.csv"))
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(tmp_path / "new-study"))
    layout, log_path = run_catch22._prepare_primary_layout("test.log", "main_shared")
    env = run_catch22._base_env(layout, log_file=log_path)
    assert "COHORT_FILE" not in env
    assert "CATCH_22_FILE" not in env
    assert Path(env["PROCESSED_DIR"]).is_relative_to(tmp_path / "new-study")
    assert env["AKI_SPLIT_CACHE_DIR"] == str(layout.results_dir / "metadata/split_cache")


def test_smoke_preserves_nonempty_root(tmp_path, monkeypatch, capsys):
    root = tmp_path / "previous"
    root.mkdir()
    evidence = root / "keep.txt"
    evidence.write_text("historical")
    monkeypatch.setenv("SMOKE_ROOT", str(root))
    assert run_catch22.main(["smoke"]) == 1
    assert evidence.read_text() == "historical"
    assert "fresh or empty" in capsys.readouterr().err


def test_smoke_sampling_preserves_each_outcomes_patient_support(tmp_path):
    import json
    import pandas as pd
    from data_preparation.smoke_trim_cohort import trim_cohort_for_smoke
    frame = pd.DataFrame({"caseid": range(40), "subjectid": range(40),
                          "aki_label": [1] * 8 + [0] * 32,
                          "y_icu_admit": [0] * 24 + [1] * 8 + [0] * 8})
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    from data_preparation.artifact_metadata import metadata_path_for
    source_flow = {"stages": [{"key": "source", "count": 40, "n_patients": 40}]}
    metadata_path_for(cohort_path).write_text(json.dumps({"outer_flow_counts": source_flow}))
    selected = trim_cohort_for_smoke(cohort_path, limit=10, outcome_names=["any_aki", "icu_admission"])
    metadata = json.loads(metadata_path_for(cohort_path).read_text())
    assert metadata["outer_flow_counts"] == source_flow
    assert metadata["diagnostic_trim"]["diagnostic"] is True
    assert metadata["diagnostic_trim"]["before_operations"] == 40
    assert metadata["diagnostic_trim"]["after_operations"] == len(selected)
    assert metadata["diagnostic_trim"]["before_patients"] == 40
    assert metadata["diagnostic_trim"]["after_patients"] == selected["subjectid"].nunique()
    for column in ("aki_label", "y_icu_admit"):
        assert selected.groupby(column)["subjectid"].nunique().min() >= 8
    assert len(selected) > 10


def test_scoped_descriptives_reject_legacy_counts(tmp_path, capsys):
    assert run_catch22.main(["descriptive", "--counts-file", str(tmp_path / "old.json")]) == 1
    assert "source-flow metadata" in capsys.readouterr().err


def test_descriptive_branches_have_distinct_artifact_names(tmp_path, monkeypatch):
    processed = tmp_path / 'processed'
    _write_prepared_fixture(processed)
    cohort = processed / 'aki_features_master_wide.csv'
    for name, value in [('PROCESSED_DIR', processed), ('RESULTS_DIR', tmp_path / 'results'),
                        ('PAPER_DIR', tmp_path / 'paper'), ('LOG_FILE', tmp_path / 'run.log')]:
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv('AKI_STORAGE_POLICY', 'off')
    calls = []
    monkeypatch.setattr(run_catch22, 'run_module', lambda module, args, **kwargs: calls.append((module, args)) or 0)
    for branch in ('non_windowed', 'windowed'):
        assert run_catch22.main(['descriptive', '--cohort-csv', str(cohort), '--branch', branch]) == 0
    flows = [args[args.index('--output-name') + 1] for module, args in calls if module == 'reporting.cohort_flow']
    assert flows == ['cohort_flow', 'cohort_flow_windowed']
    tables = [args[args.index('--output-prefix') + 1] for module, args in calls if module != 'reporting.cohort_flow']
    assert tables == ['preop_descriptives', 'missingness_table', 'preop_descriptives_windowed', 'missingness_table_windowed']


def test_multiscope_core_dry_run_defaults_to_optimized_and_preserves_fixed_budgets(tmp_path, monkeypatch, capsys):
    import json
    from model_creation import experiment_queue, config_scheduler
    untouched = tmp_path / "new-study"
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(untouched))
    for name in ("PROCESSED_DIR", "RESULTS_DIR", "PAPER_DIR", "LOG_FILE", "THREADS_PER_MODEL", "MAX_WORKERS"):
        monkeypatch.delenv(name, raising=False)
    for module in (run_catch22, experiment_queue, config_scheduler):
        monkeypatch.setattr(module, "available_cpu_count", lambda: 64)
    monkeypatch.setattr(run_catch22, "run_module", lambda *a, **k: pytest.fail("Dry run executed work"))
    monkeypatch.setattr(experiment_queue, "run_configurations", lambda *a, **k: pytest.fail("Dry run queued training"))
    assert run_catch22.main(["experiments", "--preset", "core", "--cohort-scopes",
        "icu_waveform_sensitivity", "mortality_waveform_sensitivity", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["scheduler"] == "optimized" and plan["cpu_budget"] == 60
    assert plan["pending_configurations"] == 36
    assert len(plan["configurations"]) == 36
    scope_outcomes = {(c["cohort_scope"], c["outcome"]) for c in plan["configurations"]}
    assert scope_outcomes == {("icu_waveform_sensitivity", "icu_admission"), ("mortality_waveform_sensitivity", "mortality")}
    for config in plan["configurations"]:
        resources = config["resources"]
        assert resources["max_workers"] == 5
        if config["model_type"] == "logistic_regression":
            assert resources["blas_threads"] == 4 and resources["cpu_reservation"] == 20
        elif config["model_type"] == "xgboost":
            assert resources["threads_per_model"] == 4 and resources["cpu_reservation"] == 20
        else:
            assert resources["threads_per_model"] == 12 and resources["cpu_reservation"] == 18
        assert config["n_trials"] == (20 if config["model_type"] == "logistic_regression" else 50)
        assert config["feature_set"] in run_catch22.CORE_FEATURE_SETS
    assert all(scope in plan["scope_paths"][scope]["results_dir"] for scope in plan["cohort_scopes"])
    assert not untouched.exists()


def test_mortality_scope_dry_run_has_exploratory_role_and_no_aki_requirement(tmp_path, monkeypatch, capsys):
    import json
    root = tmp_path / "absent-inputs-and-outputs"
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(root))
    monkeypatch.setattr(run_catch22, "_ensure_data_prepared", lambda *a, **k: pytest.fail("Dry run prepared data"))
    assert run_catch22.main(["experiments", "--preset", "core", "--cohort-scope",
                            "mortality_waveform_sensitivity", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["configurations"]) == 18
    assert {c["outcome"] for c in plan["configurations"]} == {"mortality"}
    assert plan["analysis_role"] == "exploratory" and plan["diagnostic"] is False
    assert plan["outer_folds"] == 5 and plan["inner_folds"] == 3 and plan["calibration_folds"] == 5
    assert not root.exists()


@pytest.mark.parametrize("arguments", [
    ["--cpu-budget", "0"], ["--max-concurrent-configs", "0"],
    ["--blas-threads", "0"], ["--rerun-completed"],
    ["--cohort-scope", "mortality_waveform_sensitivity", "--outcomes", "icu_admission"],
])
def test_queue_rejects_invalid_controls_before_writes(arguments, tmp_path, monkeypatch):
    root = tmp_path / "untouched"
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(root))
    assert run_catch22.main(["experiments", "--dry-run", *arguments]) == 1
    assert not root.exists()


@pytest.mark.parametrize("exit_code", [0, None], ids=["missing-bundles", "spawn-failure"])
def test_failed_or_incomplete_configuration_skips_reports(tmp_path, monkeypatch, exit_code):
    import json
    root = tmp_path / "new-study"
    monkeypatch.setenv("AKI_ARTIFACT_ROOT", str(root))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    for name in ("PROCESSED_DIR", "RESULTS_DIR", "PAPER_DIR", "LOG_FILE"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    def prepare(module, args, **kwargs):
        calls.append((module, args))
        assert module == "data_preparation.validate_processed_artifacts"
        _write_prepared_fixture(kwargs["env"]["PROCESSED_DIR"])
        return 0
    monkeypatch.setattr(run_catch22, "run_module", prepare)
    _mock_experiment_queue(monkeypatch, calls, verify=False, exit_code=exit_code)
    assert run_catch22.main(["experiments", "--prep", "skip", "--preset", "core", "--only-xgboost",
        "--outcomes", "icu_admission", "--branches", "non_windowed", "--feature-sets", "preop_only"]) == 1
    manifest_path = root / "results/catch22/experiments/main_shared/nested_cv/run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert manifest["execution_history"][0]["status"] == ("artifact_validation_failed" if exit_code == 0 else "failed")
    assert not any(module.startswith("reporting") or module == "results_recreation.metrics_summary" for module, _ in calls)
