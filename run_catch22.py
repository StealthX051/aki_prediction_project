from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from artifact_paths import (
    StoragePolicyError,
    build_catch22_layout,
    enforce_storage_policy,
    get_storage_policy,
    get_scoped_results_dir,
    get_scoped_paper_dir,
    scope_artifact_dir,
    refresh_repo_convenience_paths,
    resolve_log_file,
)
from data_preparation.cohort_scopes import COHORT_SCOPES, SCOPE_OUTCOMES, scope_required_columns
from runtime_resources import available_cpu_count, resource_snapshot

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_OUTCOMES = ("any_aki", "icu_admission")
DEFAULT_BRANCHES = ("non_windowed", "windowed")
SUPPORTED_MODEL_TYPES = ("logistic_regression", "xgboost", "ebm")
CORE_FEATURE_SETS = ("preop_only", "all_waveforms", "preop_and_all_waveforms")

PRIMARY_FEATURE_SETS = (
    "preop_only",
    "pleth_only",
    "ecg_only",
    "co2_only",
    "awp_only",
    "all_waveforms",
    "preop_and_all_waveforms",
)
ABLATION_FEATURE_SETS_SINGLE = (
    "preop_and_pleth",
    "preop_and_ecg",
    "preop_and_co2",
    "preop_and_awp",
)
ABLATION_FEATURE_SETS_MINUS_ONE = (
    "preop_and_all_minus_pleth",
    "preop_and_all_minus_ecg",
    "preop_and_all_minus_co2",
    "preop_and_all_minus_awp",
)
ALL_FEATURE_SETS = (
    *PRIMARY_FEATURE_SETS,
    *ABLATION_FEATURE_SETS_SINGLE,
    *ABLATION_FEATURE_SETS_MINUS_ONE,
)

REPORT_ENV_DEFAULTS = {
    "CALIBRATION_BIN_STRATEGY": "quantile",
    "CALIBRATION_N_BINS": "10",
    "CALIBRATION_SHOW_BIN_COUNTS": "true",
    "CALIBRATION_SHOW_PROB_HIST": "true",
    "CALIBRATION_SHOW_XLIM_INSET": "true",
    "CALIBRATION_MAX_COUNT_ANNOTATE": "30",
    "PLOT_PREFER_CALIBRATED": "true",
    "PLOT_N_JOBS": "-2",
    "PR_SHOW_CLASS_BALANCE": "false",
}


def _emit(message: str, log_handle) -> None:
    print(message)
    if log_handle is not None:
        log_handle.write(message + "\n")
        log_handle.flush()


def _stream_output(proc: subprocess.Popen[str], log_handle) -> int:
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_handle is not None:
            log_handle.write(line)
            log_handle.flush()
    return proc.wait()


def _run_command(command: Sequence[str], *, env: dict[str, str], log_handle) -> int:
    process = subprocess.Popen(
        list(command),
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return _stream_output(process, log_handle)


def run_module(module: str, args: Sequence[str], *, env: dict[str, str], log_handle) -> int:
    command = [sys.executable, "-m", module, *args]
    return _run_command(command, env=env, log_handle=log_handle)


def _csv_items(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    return default


def _resolve_model_types(parsed_args: argparse.Namespace) -> list[str]:
    if getattr(parsed_args, "only_xgboost", False) and getattr(parsed_args, "only_ebm", False):
        raise ValueError("Select only one model-family shortcut.")
    if getattr(parsed_args, "only_xgboost", False):
        model_types = ["xgboost"]
    elif getattr(parsed_args, "only_ebm", False):
        model_types = ["ebm"]
    else:
        model_types = _csv_items(getattr(parsed_args, "model_types", None)) or list(SUPPORTED_MODEL_TYPES)

    invalid = [model_type for model_type in model_types if model_type not in SUPPORTED_MODEL_TYPES]
    if invalid:
        raise ValueError(
            f"Unsupported model type(s): {', '.join(invalid)}. Expected: {', '.join(SUPPORTED_MODEL_TYPES)}."
        )
    return list(dict.fromkeys(model_types))


def _allowlist(raw, defaults, allowed, label: str) -> list[str]:
    if raw is None:
        return list(defaults)
    items = _csv_items(raw) if isinstance(raw, str) else [item for value in raw for item in _csv_items(value)]
    if not items or any(item not in allowed for item in items):
        raise ValueError(f"Invalid {label}; choose from {', '.join(allowed)}")
    return list(dict.fromkeys(items))


def _configuration_grid(args) -> list[dict]:
    models = _resolve_model_types(args)
    defaults = SCOPE_OUTCOMES[args.cohort_scope]
    if args.outcomes is not None and any(item not in defaults for value in args.outcomes for item in _csv_items(value)):
        raise ValueError(f"{args.cohort_scope} supports only {', '.join(defaults)}")
    outcomes = _allowlist(args.outcomes, defaults, defaults, "outcomes")
    features = _allowlist(args.feature_sets, CORE_FEATURE_SETS if args.preset == "core" else ALL_FEATURE_SETS,
                          ALL_FEATURE_SETS, "feature sets")
    branches = _allowlist(args.branches, DEFAULT_BRANCHES, DEFAULT_BRANCHES, "branches")
    return [{"model_type": model, "outcome": outcome, "branch": branch, "feature_set": feature,
             "n_trials": min(args.n_trials or (20 if model == "logistic_regression" else 50), 2) if args.smoke
                         else args.n_trials or (20 if model == "logistic_regression" else 50)}
            for model in models for outcome in outcomes for branch in branches for feature in features]


def _write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _study_identity() -> dict:
    from model_creation.validation import build_validation_fingerprint
    fingerprint = build_validation_fingerprint(
        dataset_hash="", validation_scheme="nested_cv", model_type="logistic_regression",
        outcome="any_aki", branch="non_windowed", feature_set="preop_only",
        outer_folds=5, inner_folds=3, calibration_folds=5, repeats=1, n_trials=20, threads_per_model=1,
        preserve_nan=True, base_random_state=42)
    return {key: fingerprint[key] for key in ("validation_protocol_version", "code_revision", "code_hash",
                                              "environment", "preprocessing_protocol")}


def _freeze_branch_rosters(processed_dir: Path, results_dir: Path, scope: str, branches: Iterable[str]) -> dict:
    from data_preparation.artifact_metadata import STEP_05_MERGED_ARTIFACT, read_versioned_csv, file_sha256
    from data_preparation.cohort_scopes import select_cohort, build_cohort_manifest, selected_case_hash

    rosters = {}
    for branch in dict.fromkeys(branches):
        suffix = "_windowed" if branch == "windowed" else ""
        data_path = processed_dir / f"aki_features_master_wide{suffix}.csv"
        frame, _ = read_versioned_csv(data_path, artifact_role=STEP_05_MERGED_ARTIFACT,
                                     required_columns=["caseid", "subjectid", *scope_required_columns(scope)])
        selected = select_cohort(frame, scope)
        if selected.empty:
            raise ValueError(f"No operations available for {scope}/{branch}")
        roster_path = results_dir / "metadata" / f"{branch}_roster.csv"
        roster_path.parent.mkdir(parents=True, exist_ok=True)
        if roster_path.exists():
            import pandas as pd
            if selected_case_hash(pd.read_csv(roster_path)) != selected_case_hash(selected):
                raise ValueError("Frozen operation roster changed; use a fresh study root.")
        else:
            selected[["caseid", "subjectid"]].to_csv(roster_path, index=False)
        rosters[branch] = {**build_cohort_manifest(frame, scope, branch), "path": str(roster_path),
                           "dataset_sha256": file_sha256(data_path)}
    return rosters


@contextmanager
def _exclusive_run(path: Path):
    """Prevent simultaneous writers while allowing interrupted processes to resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process is writing this run: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _base_env(layout, *, log_file: Path) -> dict[str, str]:
    env = os.environ.copy()
    # A launcher owns its structured output layout; custom per-file output
    # overrides belong to the explicitly staged module commands.
    for name in ("COHORT_FILE", "CATCH_22_FILE", "CATCH_22_ERROR_FILE", "PREOP_PROCESSED_FILE",
                 "CATCH_22_WINDOWED_FILE", "INTRAOP_WIDE_FILE", "INTRAOP_WIDE_WINDOWED_FILE",
                 "WIDE_FEATURES_FILE", "WIDE_FEATURES_WINDOWED_FILE", "COHORT_FLOW_COUNTS_FILE",
                 "MERGED_DATASET_PATH"):
        env.pop(name, None)
    env.update(
        {
            "AKI_ARTIFACT_ROOT": str(layout.artifact_root),
            "AKI_STORAGE_POLICY": get_storage_policy(),
            "DATA_DIR": str(layout.data_dir),
            "RAW_DIR": str(layout.raw_dir),
            "PROCESSED_DIR": str(layout.processed_dir),
            "RESULTS_DIR": str(layout.results_dir),
            "PAPER_DIR": str(layout.paper_dir),
            "SMOKE_ROOT": str(layout.smoke_root),
            "LOG_FILE": str(log_file),
            "FIGURES_DIR": str(layout.paper_dir / "figures"),
            "TABLES_DIR": str(layout.paper_dir / "tables"),
            "REPORTS_DIR": str(layout.paper_dir / "reports"),
            "XAI_DIR": str(layout.results_dir / "xai"),
            "AKI_SPLIT_CACHE_DIR": str(layout.results_dir / "metadata" / "split_cache"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _report_env(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    for key, value in REPORT_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    if "AKI_REPORT_CPU_BUDGET" in env:
        env["PLOT_N_JOBS"] = str(min(available_cpu_count(), int(env["AKI_REPORT_CPU_BUDGET"])))
    return env


def _open_log_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def _prepare_primary_layout(default_log_name: str, cohort_scope: str | None = None,
                            validation_scheme: str = "nested_cv") -> tuple[object, Path]:
    layout = build_catch22_layout(PROJECT_ROOT, default_log_name=default_log_name)
    if cohort_scope is not None:
        layout = replace(layout,
                         results_dir=get_scoped_results_dir(PROJECT_ROOT, cohort_scope, validation_scheme),
                         paper_dir=get_scoped_paper_dir(PROJECT_ROOT, cohort_scope, validation_scheme))
    log_file = resolve_log_file(default_log_name)
    return layout, log_file


def _prepare_smoke_layout(default_log_name: str) -> tuple[object, Path, Path, Path, Path, Path]:
    layout = build_catch22_layout(PROJECT_ROOT, default_log_name=default_log_name)
    smoke_root = layout.smoke_root
    smoke_data_dir = smoke_root / "data"
    smoke_processed_dir = smoke_data_dir / "processed"
    smoke_results_dir = smoke_root / "results" / "catch22" / "experiments"
    smoke_paper_dir = smoke_root / "results" / "catch22" / "paper"
    log_file = resolve_log_file(default_log_name, default_dir=smoke_root)
    return layout, log_file, smoke_data_dir, smoke_processed_dir, smoke_results_dir, smoke_paper_dir


def _ensure_paths(paths: dict[str, Path]) -> None:
    enforce_storage_policy(paths)
    for name, path in paths.items():
        if name.endswith("_file"):
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _run_reporting(
    results_dir: Path, parallel_backend: str, *, env: dict[str, str], log_handle,
    bootstrap: int = 5000, bootstrap_seed: int = 0, bootstrap_stratified: bool = False,
) -> int:
    _emit("--- Evaluating models and generating reports ---", log_handle)
    _emit("  > Running evaluation (results -> results directory)...", log_handle)
    manifest_path = results_dir / "run_manifest.json"
    report_workers = min(available_cpu_count(), int(env.get("AKI_REPORT_CPU_BUDGET", available_cpu_count())))
    if manifest_path.exists():
        report_workers = min(report_workers, max(1, len(json.loads(manifest_path.read_text())["configurations"])))
    metrics_args = [
        "--results-dir",
        str(results_dir),
        "--delta-mode",
        "reference",
        "--reference-feature-set",
        "preop_only",
        "--parallel-backend",
        parallel_backend,
        "--n-jobs",
        str(report_workers),
        "--bootstrap-timeout",
        "1800",
        "--bootstrap-max-retries",
        "2",
        "--bootstrap", str(bootstrap),
        "--bootstrap-seed", str(bootstrap_seed),
        "--bootstrap-stratified" if bootstrap_stratified else "--no-bootstrap-stratified",
    ]
    if manifest_path.exists():
        metrics_args.extend(["--strict", "--run-manifest", str(manifest_path)])
    if run_module("results_recreation.metrics_summary", metrics_args, env=env, log_handle=log_handle) != 0:
        _emit("Evaluation failed.", log_handle)
        return 1

    _emit("  > Building reports from standardized artifacts...", log_handle)
    report_args = ["--strict", "--run-manifest", str(manifest_path)] if manifest_path.exists() else []
    if run_module("reporting.make_report", report_args, env=_report_env(env), log_handle=log_handle) != 0:
        _emit("Report generation failed.", log_handle)
        return 1
    return 0


def _validate_processed_artifacts(*, env: dict[str, str], log_handle) -> int:
    result = run_module("data_preparation.validate_processed_artifacts", [], env=env, log_handle=log_handle)
    if result != 0:
        _emit("Existing processed data failed schema validation. Rebuild with --prep force.", log_handle)
    return result


def _ensure_data_prepared(
    prep_mode: str,
    *,
    env: dict[str, str],
    processed_dir: Path,
    log_handle,
) -> int:
    full_file = processed_dir / "aki_features_master_wide.csv"
    windowed_file = processed_dir / "aki_features_master_wide_windowed.csv"

    if prep_mode == "skip":
        _emit("Skipping data prep (prep=skip).", log_handle)
        return _validate_processed_artifacts(env=env, log_handle=log_handle)

    if prep_mode != "force" and full_file.exists() and windowed_file.exists():
        _emit(f"Reusing existing processed data at {processed_dir} (use --prep force to rebuild).", log_handle)
        return _validate_processed_artifacts(env=env, log_handle=log_handle)

    _emit("--- Running data preparation (steps 01-05) ---", log_handle)
    if prep_mode == "force" and processed_dir.exists():
        if any(processed_dir.iterdir()):
            raise ValueError("--prep force requires an empty processed directory; select a fresh study root to preserve existing artifacts.")

    prep_env = dict(env)
    prep_env.update(GENERATE_FULL_FEATURES="True", GENERATE_WINDOWED_FEATURES="True")
    pipeline = (
        ("data_preparation.step_01_cohort_construction", "Step 01 failed"),
        ("data_preparation.step_02_catch_22", "Step 02 failed"),
        ("data_preparation.step_03_preop_prep", "Step 03 failed"),
        ("data_preparation.step_04_intraop_prep", "Step 04 failed"),
        ("data_preparation.step_05_data_merge", "Step 05 failed"),
    )
    for module, failure_message in pipeline:
        if run_module(module, [], env=prep_env, log_handle=log_handle) != 0:
            _emit(failure_message, log_handle)
            return 1
    return _validate_processed_artifacts(env=prep_env, log_handle=log_handle)


def _run_experiments(args: argparse.Namespace) -> int:
    from model_creation.experiment_queue import execute
    return execute(args)


def _run_smoke(args: argparse.Namespace) -> int:
    try:
        model_types = _resolve_model_types(args)
        allowed = SCOPE_OUTCOMES[args.cohort_scope]
        args.outcomes = _allowlist(args.outcomes, allowed, allowed, "outcomes")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    layout, log_file, smoke_data_dir, smoke_processed_dir, smoke_results_dir, smoke_paper_dir = _prepare_smoke_layout(
        "smoke_test.log"
    )
    smoke_results_dir = scope_artifact_dir(smoke_results_dir, args.cohort_scope, "holdout")
    smoke_paper_dir = scope_artifact_dir(smoke_paper_dir, args.cohort_scope, "holdout")
    if layout.smoke_root.exists() and any(layout.smoke_root.iterdir()):
        raise ValueError("Smoke requires a fresh or empty SMOKE_ROOT; existing outputs are preserved.")
    try:
        _ensure_paths(
            {
                "smoke_root": layout.smoke_root,
                "smoke_processed_dir": smoke_processed_dir,
                "smoke_results_dir": smoke_results_dir,
                "smoke_paper_dir": smoke_paper_dir,
                "log_file": log_file,
            }
        )
    except StoragePolicyError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.refresh_links:
        refresh_repo_convenience_paths(PROJECT_ROOT, processed_dir=smoke_processed_dir,
                                       results_dir=smoke_results_dir, paper_dir=smoke_paper_dir)

    raw_source_dir = Path(args.raw_source_dir).expanduser().resolve(strict=False)
    env = _base_env(layout, log_file=log_file)
    env.update(
        {
            "DATA_DIR": str(smoke_data_dir),
            "PROCESSED_DIR": str(smoke_processed_dir),
            "RAW_DIR": str(raw_source_dir),
            "INPUT_FILE": str(raw_source_dir / "clinical_data.csv"),
            "LAB_DATA_FILE": str(raw_source_dir / "lab_data.csv"),
            "RESULTS_DIR": str(smoke_results_dir),
            "PAPER_DIR": str(smoke_paper_dir),
            "FIGURES_DIR": str(smoke_paper_dir / "figures"),
            "TABLES_DIR": str(smoke_paper_dir / "tables"),
            "REPORTS_DIR": str(smoke_paper_dir / "reports"),
            "XAI_DIR": str(smoke_results_dir / "xai"),
            "AKI_SPLIT_CACHE_DIR": str(smoke_results_dir / "metadata" / "split_cache"),
            "SMOKE_ROOT": str(layout.smoke_root),
            "GENERATE_FULL_FEATURES": "True",
            "GENERATE_WINDOWED_FEATURES": "True",
            "SMOKE_OUTCOMES": ",".join(args.outcomes),
            "COHORT_SCOPE": args.cohort_scope,
            "VALIDATION_SCHEME": "holdout",
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "2",
        }
    )

    cohort_path = smoke_processed_dir / "aki_pleth_ecg_co2_awp.csv"
    with _open_log_file(log_file) as log_handle:
        _emit("=== Starting real-data smoke test ===", log_handle)
        _emit(f"Smoke root: {layout.smoke_root}", log_handle)
        _emit(f"Using raw data from: {raw_source_dir}", log_handle)
        _emit(f"Python runner: {sys.executable}", log_handle)

        pipeline = (
            ("--- Step 01: Cohort construction ---", "data_preparation.step_01_cohort_construction", []),
            (
                f"--- Trimming cohort to {args.case_limit} cases ---",
                "data_preparation.smoke_trim_cohort",
                [
                    "--cohort-path",
                    str(cohort_path),
                    "--limit",
                    str(args.case_limit),
                    "--outcomes",
                    ",".join(args.outcomes),
                    "--cohort-scope", args.cohort_scope,
                ],
            ),
            ("--- Step 02: Catch22 feature extraction ---", "data_preparation.step_02_catch_22", []),
            ("--- Step 03: Preoperative prep & split ---", "data_preparation.step_03_preop_prep", []),
            ("--- Step 04: Intraoperative prep ---", "data_preparation.step_04_intraop_prep", []),
            ("--- Step 05: Data merge ---", "data_preparation.step_05_data_merge", []),
        )

        for banner, module, module_args in pipeline:
            _emit(banner, log_handle)
            if run_module(module, module_args, env=env, log_handle=log_handle) != 0:
                return 1

        configurations = [{"model_type": model, "outcome": outcome, "branch": "windowed", "feature_set": feature,
                           "n_trials": min(args.hpo_trials, 2)}
                          for model in model_types for outcome in args.outcomes for feature in args.feature_sets]
        branch_rosters = _freeze_branch_rosters(smoke_processed_dir, smoke_results_dir, args.cohort_scope, ["windowed"])
        study_identity = _study_identity()
        reporting_policy = {"bootstrap_replicates": 100, "bootstrap_seed": 0, "bootstrap_stratified": False}
        _write_manifest(smoke_results_dir / "run_manifest.json", {
            "cohort_scope": args.cohort_scope, "validation_scheme": "holdout", "diagnostic": True,
            "configurations": configurations, "branch_rosters": branch_rosters, "study_identity": study_identity,
            "reporting": reporting_policy, "status": "running"})
        for config in configurations:
            outcome, model_type = config["outcome"], config["model_type"]
            _emit(f"--- Step 07: Holdout validation (smoke, {outcome}, {model_type}) ---", log_handle)
            validation_args = [
                "--outcome",
                outcome,
                "--branch",
                "windowed",
                "--feature_set",
                config["feature_set"],
                "--model_type",
                model_type,
                "--validation-scheme",
                "holdout",
                "--max-workers",
                "1",
                "--threads-per-model",
                "2",
                "--n-trials",
                str(config["n_trials"]),
                "--smoke_test",
                "--diagnostic", "--cohort-scope", args.cohort_scope,
            ]
            if run_module("model_creation.step_07_train_evaluate", validation_args, env=env, log_handle=log_handle) != 0:
                return 1

        _write_manifest(smoke_results_dir / "run_manifest.json", {
            "cohort_scope": args.cohort_scope, "validation_scheme": "holdout", "diagnostic": True,
            "configurations": configurations, "branch_rosters": branch_rosters, "study_identity": study_identity,
            "reporting": reporting_policy, "status": "models_complete"})
        if _run_reporting(smoke_results_dir, args.parallel_backend, env=env, log_handle=log_handle, bootstrap=100) != 0:
            saved = json.loads((smoke_results_dir / "run_manifest.json").read_text())
            saved["status"] = "reporting_failed"
            _write_manifest(smoke_results_dir / "run_manifest.json", saved)
            return 1
        saved = json.loads((smoke_results_dir / "run_manifest.json").read_text())
        saved["status"] = "validated"
        _write_manifest(smoke_results_dir / "run_manifest.json", saved)

        _emit("=== Smoke test complete ===", log_handle)
        _emit(f"Artifacts written under {layout.smoke_root}", log_handle)
    return 0


def _run_descriptive(args: argparse.Namespace) -> int:
    if args.counts_file is not None:
        print("Scoped descriptives require source-flow metadata from --cohort-csv. Render legacy counts with python -m reporting.cohort_flow --counts-file PATH.", file=sys.stderr)
        return 1
    layout, log_file = _prepare_primary_layout("descriptive_figures.log", args.cohort_scope, args.validation_scheme)
    try:
        _ensure_paths(
            {
                "processed_dir": layout.processed_dir,
                "results_dir": layout.results_dir,
                "paper_dir": layout.paper_dir,
                "log_file": log_file,
            }
        )
    except StoragePolicyError as exc:
        print(exc, file=sys.stderr)
        return 1

    suffix = "" if args.branch == "non_windowed" else "_windowed"
    cohort_csv = Path(args.cohort_csv).expanduser().resolve(strict=False)
    merged_name = "aki_features_master_wide_windowed.csv" if args.branch == "windowed" else "aki_features_master_wide.csv"
    merged_dataset = Path(args.merged_dataset or layout.processed_dir / merged_name).expanduser().resolve(strict=False)
    display_dict = Path(args.display_dictionary).expanduser().resolve(strict=False)
    processed_preop = Path(args.processed_preop).expanduser().resolve(strict=False)

    required_paths = (
        cohort_csv,
        merged_dataset,
        display_dict,
    )
    with _open_log_file(log_file) as log_handle:
        env = _base_env(layout, log_file=log_file)
        env.update(COHORT_SCOPE=args.cohort_scope, VALIDATION_SCHEME=args.validation_scheme)
        for path in required_paths:
            if not path.exists():
                _emit(f"Required file not found: {path}", log_handle)
                _emit(
                    "Override via --cohort-csv/--merged-dataset/--display-dictionary or regenerate artifacts.",
                    log_handle,
                )
                return 1

        if not processed_preop.exists():
            _emit(
                f"Warning: Processed preop dataset not found at {processed_preop}; continuous stats will use raw cohort.",
                log_handle,
            )
        else:
            _emit(f"Using processed preop dataset for continuous features: {processed_preop}", log_handle)

        import pandas as pd
        from data_preparation.cohort_scopes import select_cohort
        roster = select_cohort(pd.read_csv(merged_dataset), args.cohort_scope)
        roster_path = layout.paper_dir / "metadata" / f"{args.branch}_roster.csv"
        roster_path.parent.mkdir(parents=True, exist_ok=True)
        roster[["caseid", "subjectid"]].to_csv(roster_path, index=False)
        cohort_args = ["--cohort-scope", args.cohort_scope, "--branch-roster", str(roster_path)]

        _emit("[1/3] Generating preoperative descriptive table...", log_handle)
        if (
            run_module(
                "reporting.preop_descriptives",
                [
                    "--dataset",
                    str(cohort_csv),
                    "--processed-dataset",
                    str(processed_preop),
                    "--display-dictionary",
                    str(display_dict),
                    "--output-prefix",
                    "preop_descriptives" + suffix,
                ] + cohort_args,
                env=env,
                log_handle=log_handle,
            )
            != 0
        ):
            return 1

        _emit("[2/3] Rendering cohort flow diagram...", log_handle)
        if (
            run_module(
                "reporting.cohort_flow",
                [
                    "--dataset", str(cohort_csv),
                    "--display-dictionary",
                    str(display_dict),
                    "--output-name",
                    "cohort_flow" + suffix,
                ] + cohort_args,
                env=env,
                log_handle=log_handle,
            )
            != 0
        ):
            return 1

        _emit("[3/3] Computing missingness table...", log_handle)
        if (
            run_module(
                "reporting.missingness_table",
                [
                    "--dataset",
                    str(merged_dataset),
                    "--display-dictionary",
                    str(display_dict),
                    "--output-prefix",
                    "missingness_table" + suffix,
                ] + cohort_args,
                env=env,
                log_handle=log_handle,
            )
            != 0
        ):
            return 1

        _emit("Done. Outputs:", log_handle)
        _emit(f"  - Demographics:    {layout.paper_dir / 'tables' / ('preop_descriptives' + suffix)}.*", log_handle)
        _emit(f"  - Cohort flow:     {layout.paper_dir / 'figures' / ('cohort_flow' + suffix)}.*", log_handle)
        _emit(f"  - Missingness:     {layout.paper_dir / 'tables' / ('missingness_table' + suffix)}.*", log_handle)
    return 0


def _add_model_type_arguments(parser: argparse.ArgumentParser, *, default_model_types: str) -> None:
    parser.add_argument(
        "--model-types",
        default=default_model_types,
        help=f"Comma-separated model types (default: {default_model_types}).",
    )
    parser.add_argument("--only-xgboost", action="store_true", help="Shortcut for --model-types xgboost.")
    parser.add_argument("--only-ebm", action="store_true", help="Shortcut for --model-types ebm.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Catch22 pipeline launcher.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    experiments = subparsers.add_parser("experiments", help="Run the full Catch22 experiment grid.")
    _add_model_type_arguments(experiments, default_model_types=",".join(SUPPORTED_MODEL_TYPES))
    experiments.add_argument("--preset", choices=("full", "core"), default="full")
    experiments.add_argument("--feature-sets", nargs="+", help="Feature allowlist; overrides the preset.")
    experiments.add_argument("--outcomes", nargs="+", help="Outcome allowlist.")
    experiments.add_argument("--branches", nargs="+", help="Processing branch allowlist.")
    scopes = experiments.add_mutually_exclusive_group()
    scopes.add_argument("--cohort-scope", choices=COHORT_SCOPES, default=None)
    scopes.add_argument("--cohort-scopes", choices=COHORT_SCOPES, nargs="+")
    experiments.add_argument("--scheduler", choices=("optimized", "legacy"), default="optimized")
    experiments.add_argument("--admission-policy", choices=("adaptive", "reserved"), default="adaptive",
                             help="Adaptive targets measured CPU use; reserved enforces summed peak reservations.")
    experiments.add_argument("--cpu-budget", type=int)
    experiments.add_argument("--max-concurrent-configs", type=int)
    experiments.add_argument("--reuse-from", type=Path, action="append", default=[])
    experiments.add_argument("--rerun-completed", action="store_true", help="Explicit fresh-root rerun; requires --no-resume.")
    experiments.add_argument("--no-profile-resources", dest="profile_resources", action="store_false", default=True)
    experiments.add_argument("--dry-run", action="store_true", help="Print configurations without filesystem changes.")
    experiments.add_argument("--diagnostic", action="store_true", help="Label reduced validation as diagnostic.")
    experiments.add_argument("--report-only", action="store_true", help="Regenerate from saved predictions without fitting.")
    experiments.add_argument("--skip-report", action="store_true", help="Save validation artifacts without reporting yet.")
    experiments.add_argument("--refresh-links", action="store_true", help="Refresh repository convenience symlinks explicitly.")
    experiments.add_argument(
        "--prep",
        choices=("auto", "force", "skip"),
        default=os.getenv("PREP_MODE", "auto"),
        help="Data preparation mode: auto, force, or skip.",
    )
    experiments.add_argument(
        "--smoke",
        action="store_true",
        default=_env_flag("SMOKE_TEST_FLAG", False),
        help="Pass --smoke_test into training/evaluation.",
    )
    experiments.add_argument(
        "--validation-scheme",
        choices=("nested_cv", "holdout"),
        default=os.getenv("VALIDATION_SCHEME", "nested_cv"),
        help="Validation scheme for step_07_train_evaluate.",
    )
    experiments.add_argument("--outer-folds", type=int, default=_env_int("OUTER_FOLDS", 5))
    experiments.add_argument("--inner-folds", type=int, default=_env_int("INNER_FOLDS", 3), help="Patient-grouped HPO folds.")
    experiments.add_argument("--calibration-folds", type=int, default=_env_int("CALIBRATION_FOLDS", 5),
                             help="Patient-grouped training-only calibration folds.")
    experiments.add_argument("--repeats", type=int, default=_env_int("REPEATS", 1))
    experiments.add_argument("--max-workers", type=int, default=_env_int("MAX_WORKERS", min(5, available_cpu_count())))
    experiments.add_argument("--threads-per-model", type=int, default=int(os.environ["THREADS_PER_MODEL"]) if os.getenv("THREADS_PER_MODEL") else None)
    experiments.add_argument("--blas-threads", type=int, default=None,
                             help="Bounded BLAS threads for LR; other families use one BLAS thread.")
    experiments.add_argument("--n-trials", type=int, default=int(os.environ["N_TRIALS"]) if os.getenv("N_TRIALS") else None,
                             help="HPO trials; defaults to 20 for LR and 50 for XGBoost/EBM.")
    experiments.add_argument("--bootstrap", type=int, default=5000,
                             help="Patient-cluster reporting resamples (default: 5000; smoke capped at 100).")
    experiments.add_argument("--bootstrap-seed", type=int, default=0)
    experiments.add_argument("--bootstrap-stratified", action="store_true", default=False,
                             help="Condition reporting resamples on patient outcome strata (default: unstratified).")
    experiments.add_argument("--no-bootstrap-stratified", action="store_false", dest="bootstrap_stratified")
    experiments.add_argument(
        "--parallel-backend",
        choices=("threads", "processes"),
        default=os.getenv("PARALLEL_BACKEND", "processes"),
        help="Bootstrap parallel backend for metrics_summary.",
    )
    resume_group = experiments.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="Resume matching fold checkpoints.")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume support.")
    experiments.set_defaults(
        resume=_env_flag("RESUME_FLAG", True),
        handler=_run_experiments,
    )
    experiments.add_argument(
        "--save-final-refit",
        dest="save_final_refit",
        action="store_true",
        default=_env_flag("SAVE_FINAL_REFIT_FLAG", False),
        help="Save an optional full-data refit bundle after validation.",
    )

    smoke = subparsers.add_parser("smoke", help="Run an isolated real-data smoke test.")
    _add_model_type_arguments(smoke, default_model_types=",".join(SUPPORTED_MODEL_TYPES))
    smoke.add_argument("--cohort-scope", choices=COHORT_SCOPES, default="main_shared")
    smoke.add_argument("--feature-sets", nargs="+", choices=ALL_FEATURE_SETS, default=list(CORE_FEATURE_SETS))
    smoke.add_argument("--refresh-links", action="store_true")
    smoke.add_argument("--case-limit", type=int, default=_env_int("CASE_LIMIT", 40))
    smoke.add_argument("--hpo-trials", type=int, default=_env_int("HPO_TRIALS", 2))
    smoke.add_argument(
        "--outcomes",
        nargs="+",
        default=_csv_items(os.getenv("SMOKE_OUTCOMES")) or None,
        help="Smoke outcomes to run (default: any_aki icu_admission).",
    )
    smoke.add_argument(
        "--parallel-backend",
        choices=("threads", "processes"),
        default=os.getenv("PARALLEL_BACKEND", "processes"),
        help="Bootstrap parallel backend for metrics_summary.",
    )
    smoke.add_argument(
        "--raw-source-dir",
        default=os.getenv("RAW_SOURCE_DIR", str(PROJECT_ROOT / "data" / "raw")),
        help="Raw data source for the smoke run.",
    )
    smoke.set_defaults(handler=_run_smoke)

    descriptive = subparsers.add_parser("descriptive", help="Generate descriptive figures and tables.")
    descriptive.add_argument("--cohort-scope", choices=COHORT_SCOPES, default="main_shared")
    descriptive.add_argument("--validation-scheme", choices=("nested_cv", "holdout"), default="nested_cv")
    descriptive.add_argument("--branch", choices=DEFAULT_BRANCHES, default="non_windowed")
    layout = build_catch22_layout(PROJECT_ROOT, default_log_name="descriptive_figures.log")
    descriptive.add_argument(
        "--display-dictionary",
        default=os.getenv("DISPLAY_DICT", str(PROJECT_ROOT / "metadata" / "display_dictionary.json")),
        help="Path to metadata/display_dictionary.json.",
    )
    descriptive.add_argument(
        "--cohort-csv",
        default=os.getenv("COHORT_CSV", str(layout.processed_dir / "aki_pleth_ecg_co2_awp.csv")),
        help="Path to the saved cohort CSV.",
    )
    descriptive.add_argument(
        "--merged-dataset",
        default=os.getenv("MERGED_DATASET"),
        help="Path to the merged modeling dataset.",
    )
    descriptive.add_argument(
        "--counts-file",
        default=os.getenv("COUNTS_FILE"),
        help="Legacy option rejected for scoped descriptives; use reporting.cohort_flow --counts-file directly.",
    )
    descriptive.add_argument(
        "--processed-preop",
        default=os.getenv("PROCESSED_PREOP", str(layout.processed_dir / "aki_preop_processed.csv")),
        help="Optional processed preop dataset for continuous statistics.",
    )
    descriptive.set_defaults(handler=_run_descriptive)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "experiments":
            if args.max_workers < 1 or (args.threads_per_model is not None and args.threads_per_model < 1):
                raise ValueError("Worker and thread counts must be positive")
            cpus = available_cpu_count()
            if args.cpu_budget is not None and not 1 <= args.cpu_budget <= cpus:
                raise ValueError("CPU budget must be positive and within available affinity/quota")
            if args.max_concurrent_configs is not None and args.max_concurrent_configs < 1:
                raise ValueError("Maximum concurrent configurations must be positive")
            if args.blas_threads is not None and args.blas_threads < 1:
                raise ValueError("LR BLAS threads must be positive")
            if args.n_trials is not None and args.n_trials < 1:
                raise ValueError("HPO trials must be positive")
            if args.bootstrap < 0 or args.bootstrap_seed < 0:
                raise ValueError("Bootstrap count and seed must be nonnegative")
            if min(args.outer_folds, args.inner_folds, args.calibration_folds) < 2:
                raise ValueError("Outer, HPO, and calibration fold counts must each be at least two")
            if args.repeats != 1:
                raise ValueError("The supported validation protocol requires one repeat")
        return args.handler(args)
    except (ValueError, StoragePolicyError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
