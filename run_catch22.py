from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

from artifact_paths import (
    StoragePolicyError,
    build_catch22_layout,
    enforce_storage_policy,
    get_storage_policy,
    refresh_repo_convenience_paths,
    resolve_log_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_OUTCOMES = ("any_aki", "icu_admission")
DEFAULT_BRANCHES = ("non_windowed", "windowed")
SUPPORTED_MODEL_TYPES = ("xgboost", "ebm")

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
    return model_types


def _base_env(layout, *, log_file: Path) -> dict[str, str]:
    env = os.environ.copy()
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
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _report_env(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    for key, value in REPORT_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    return env


def _open_log_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def _prepare_primary_layout(default_log_name: str) -> tuple[object, Path]:
    layout = build_catch22_layout(PROJECT_ROOT, default_log_name=default_log_name)
    log_file = resolve_log_file(default_log_name)
    return layout, log_file


def _prepare_smoke_layout(default_log_name: str) -> tuple[object, Path, Path, Path, Path]:
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
    for path in paths.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _run_reporting(results_dir: Path, parallel_backend: str, *, env: dict[str, str], log_handle) -> int:
    _emit("--- Evaluating models and generating reports ---", log_handle)
    _emit("  > Running evaluation (results -> results directory)...", log_handle)
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
        "-1",
        "--bootstrap-timeout",
        "1800",
        "--bootstrap-max-retries",
        "2",
    ]
    if run_module("results_recreation.metrics_summary", metrics_args, env=env, log_handle=log_handle) != 0:
        _emit("Evaluation failed.", log_handle)
        return 1

    _emit("  > Building reports from standardized artifacts...", log_handle)
    if run_module("reporting.make_report", [], env=_report_env(env), log_handle=log_handle) != 0:
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
        _emit(f"Forcing rebuild: removing existing processed directory at {processed_dir}", log_handle)
        shutil.rmtree(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)

    prep_env = dict(env)
    prep_env["GENERATE_WINDOWED_FEATURES"] = "True"
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
    return 0


def _run_experiments(args: argparse.Namespace) -> int:
    try:
        model_types = _resolve_model_types(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    layout, log_file = _prepare_primary_layout("experiment_log.txt")
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

    refresh_repo_convenience_paths(
        PROJECT_ROOT,
        processed_dir=layout.processed_dir,
        results_dir=layout.results_dir,
        paper_dir=layout.paper_dir,
    )

    env = _base_env(layout, log_file=log_file)
    with _open_log_file(log_file) as log_handle:
        _emit("Starting Experiments...", log_handle)
        _emit(
            f"Prep mode: {args.prep} | Models: {' '.join(model_types)} | Results dir: {layout.results_dir} | "
            f"Validation: {args.validation_scheme}",
            log_handle,
        )
        _emit(f"Python runner: {sys.executable}", log_handle)

        if _ensure_data_prepared(args.prep, env=env, processed_dir=layout.processed_dir, log_handle=log_handle) != 0:
            return 1

        successful_validation_runs = 0
        failed_validation_runs = 0
        failed_configs: list[str] = []

        for model_type in model_types:
            for outcome in DEFAULT_OUTCOMES:
                for branch in DEFAULT_BRANCHES:
                    for feature_set in ALL_FEATURE_SETS:
                        _emit("----------------------------------------------------------------", log_handle)
                        _emit(
                            f"Running: Model={model_type} | Outcome={outcome} | Branch={branch} | "
                            f"FeatureSet={feature_set}",
                            log_handle,
                        )
                        _emit("----------------------------------------------------------------", log_handle)
                        _emit(f"  > Starting Validation run for model_type={model_type}...", log_handle)

                        validation_args = [
                            "--outcome",
                            outcome,
                            "--branch",
                            branch,
                            "--feature_set",
                            feature_set,
                            "--model_type",
                            model_type,
                            "--validation-scheme",
                            args.validation_scheme,
                            "--outer-folds",
                            str(args.outer_folds),
                            "--inner-folds",
                            str(args.inner_folds),
                            "--repeats",
                            str(args.repeats),
                            "--max-workers",
                            str(args.max_workers),
                            "--threads-per-model",
                            str(args.threads_per_model),
                            "--n-trials",
                            str(args.n_trials),
                        ]
                        if model_type == "ebm":
                            validation_args.append("--export_ebm_explanations")
                        if args.resume:
                            validation_args.append("--resume")
                        if args.save_final_refit:
                            validation_args.append("--save-final-refit")
                        if args.smoke:
                            validation_args.append("--smoke_test")

                        if (
                            run_module(
                                "model_creation.step_07_train_evaluate",
                                validation_args,
                                env=env,
                                log_handle=log_handle,
                            )
                            == 0
                        ):
                            _emit(f"  > Validation run complete for model_type={model_type}.", log_handle)
                            successful_validation_runs += 1
                        else:
                            _emit(f"  > Validation run FAILED for model_type={model_type}.", log_handle)
                            failed_validation_runs += 1
                            failed_configs.append(
                                f"model={model_type} outcome={outcome} branch={branch} feature_set={feature_set}"
                            )

        if failed_validation_runs > 0:
            _emit("--- Validation completed with failures; skipping metrics/report generation ---", log_handle)
            for failed_config in failed_configs:
                _emit(f"  > FAILED: {failed_config}", log_handle)
            _emit(
                f"Launcher finished with {failed_validation_runs} failed validation configuration(s).",
                log_handle,
            )
            return 1

        if successful_validation_runs == 0:
            _emit("--- No validation artifacts were produced during this run; skipping metrics/report generation ---", log_handle)
            return 1

        if _run_reporting(layout.results_dir, args.parallel_backend, env=env, log_handle=log_handle) != 0:
            return 1

        _emit("All experiments finished.", log_handle)
    return 0


def _run_smoke(args: argparse.Namespace) -> int:
    try:
        model_types = _resolve_model_types(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    layout, log_file, smoke_data_dir, smoke_processed_dir, smoke_results_dir, smoke_paper_dir = _prepare_smoke_layout(
        "smoke_test.log"
    )
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

    if layout.smoke_root.exists():
        shutil.rmtree(layout.smoke_root)

    smoke_processed_dir.mkdir(parents=True, exist_ok=True)
    smoke_results_dir.mkdir(parents=True, exist_ok=True)
    smoke_paper_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    refresh_repo_convenience_paths(
        PROJECT_ROOT,
        processed_dir=smoke_processed_dir,
        results_dir=smoke_results_dir,
        paper_dir=smoke_paper_dir,
    )

    raw_source_dir = Path(args.raw_source_dir).expanduser().resolve(strict=False)
    env = _base_env(layout, log_file=log_file)
    env.update(
        {
            "DATA_DIR": str(smoke_data_dir),
            "PROCESSED_DIR": str(smoke_processed_dir),
            "RAW_DIR": str(raw_source_dir),
            "RESULTS_DIR": str(smoke_results_dir),
            "PAPER_DIR": str(smoke_paper_dir),
            "SMOKE_ROOT": str(layout.smoke_root),
            "GENERATE_WINDOWED_FEATURES": "True",
            "SMOKE_OUTCOMES": ",".join(args.outcomes),
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

        for outcome in args.outcomes:
            for model_type in model_types:
                _emit(f"--- Step 07: Holdout validation (smoke, {outcome}, {model_type}) ---", log_handle)
                validation_args = [
                    "--outcome",
                    outcome,
                    "--branch",
                    "windowed",
                    "--feature_set",
                    "all_waveforms",
                    "--model_type",
                    model_type,
                    "--validation-scheme",
                    "holdout",
                    "--max-workers",
                    "1",
                    "--threads-per-model",
                    "2",
                    "--n-trials",
                    str(args.hpo_trials),
                    "--smoke_test",
                ]
                if run_module("model_creation.step_07_train_evaluate", validation_args, env=env, log_handle=log_handle) != 0:
                    return 1

        if _run_reporting(smoke_results_dir, args.parallel_backend, env=env, log_handle=log_handle) != 0:
            return 1

        _emit("=== Smoke test complete ===", log_handle)
        _emit(f"Artifacts written under {layout.smoke_root}", log_handle)
    return 0


def _run_descriptive(args: argparse.Namespace) -> int:
    layout, log_file = _prepare_primary_layout("descriptive_figures.log")
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

    refresh_repo_convenience_paths(
        PROJECT_ROOT,
        processed_dir=layout.processed_dir,
        results_dir=layout.results_dir,
        paper_dir=layout.paper_dir,
    )

    cohort_csv = Path(args.cohort_csv).expanduser().resolve(strict=False)
    merged_dataset = Path(args.merged_dataset).expanduser().resolve(strict=False)
    display_dict = Path(args.display_dictionary).expanduser().resolve(strict=False)
    counts_file = Path(args.counts_file).expanduser().resolve(strict=False)
    processed_preop = Path(args.processed_preop).expanduser().resolve(strict=False)

    required_paths = (
        cohort_csv,
        merged_dataset,
        display_dict,
        counts_file,
    )
    with _open_log_file(log_file) as log_handle:
        env = _base_env(layout, log_file=log_file)
        for path in required_paths:
            if not path.exists():
                _emit(f"Required file not found: {path}", log_handle)
                _emit(
                    "Override via --cohort-csv/--merged-dataset/--display-dictionary/--counts-file or regenerate artifacts.",
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
                    "preop_descriptives",
                ],
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
                    "--counts-file",
                    str(counts_file),
                    "--display-dictionary",
                    str(display_dict),
                    "--output-name",
                    "cohort_flow",
                ],
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
                    "missingness_table",
                ],
                env=env,
                log_handle=log_handle,
            )
            != 0
        ):
            return 1

        _emit("Done. Outputs:", log_handle)
        _emit(f"  - Demographics:    {layout.paper_dir / 'tables' / 'preop_descriptives'}.*", log_handle)
        _emit(f"  - Cohort flow:     {layout.paper_dir / 'figures' / 'cohort_flow'}.*", log_handle)
        _emit(f"  - Missingness:     {layout.paper_dir / 'tables' / 'missingness_table'}.*", log_handle)
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
    _add_model_type_arguments(experiments, default_model_types="xgboost,ebm")
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
    experiments.add_argument("--inner-folds", type=int, default=_env_int("INNER_FOLDS", 5))
    experiments.add_argument("--repeats", type=int, default=_env_int("REPEATS", 1))
    experiments.add_argument("--max-workers", type=int, default=_env_int("MAX_WORKERS", 4))
    experiments.add_argument("--threads-per-model", type=int, default=_env_int("THREADS_PER_MODEL", 8))
    experiments.add_argument("--n-trials", type=int, default=_env_int("N_TRIALS", 100))
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
    _add_model_type_arguments(smoke, default_model_types="xgboost")
    smoke.add_argument("--case-limit", type=int, default=_env_int("CASE_LIMIT", 10))
    smoke.add_argument("--hpo-trials", type=int, default=_env_int("HPO_TRIALS", 2))
    smoke.add_argument(
        "--outcomes",
        nargs="+",
        default=_csv_items(os.getenv("SMOKE_OUTCOMES", "any_aki,icu_admission")) or list(DEFAULT_OUTCOMES),
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
        default=os.getenv("MERGED_DATASET", str(layout.processed_dir / "aki_features_master_wide.csv")),
        help="Path to the merged modeling dataset.",
    )
    descriptive.add_argument(
        "--counts-file",
        default=os.getenv("COUNTS_FILE", str(layout.paper_dir / "metadata" / "cohort_flow_counts.json")),
        help="Path to the cohort-flow counts JSON.",
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
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
