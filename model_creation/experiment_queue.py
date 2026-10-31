"""Compose existing training/reporting entrypoints into a bounded study queue."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from model_creation.config_scheduler import ConfigJob, default_cpu_budget, run_configurations
from runtime_resources import available_cpu_count, resource_snapshot


def resource_plan(args, family: str) -> dict:
    budget = args.cpu_budget or default_cpu_budget()
    native = args.threads_per_model or (min(4, available_cpu_count()) if family == "xgboost" else min(12, available_cpu_count()))
    blas = (args.blas_threads or (1 if args.scheduler == "legacy" else 4)) if family == "logistic_regression" else 1
    if args.scheduler == "legacy" and args.threads_per_model is None:
        native = max(1, available_cpu_count() // args.max_workers)
    if native > available_cpu_count() or blas > native:
        raise ValueError("Fixed estimator thread settings exceed available CPUs or per-model budget")
    if family == "ebm":
        workers = min(args.max_workers, budget - native - 1)
        reservation = native + workers + 1
    else:
        workers = min(args.max_workers, budget // (blas if family == "logistic_regression" else native))
        reservation = workers * (blas if family == "logistic_regression" else native)
    if workers < 1:
        raise ValueError(f"CPU budget cannot support fixed {family} settings; supply an explicit thread override")
    return {"max_workers": workers, "threads_per_model": native,
            "blas_threads": blas, "cpu_reservation": reservation}


def _assert_protected_paths(contexts: dict, catalogs: list[dict]) -> None:
    """Reject output paths anywhere inside an imported study, including subroots."""
    for catalog in catalogs:
        source_results = Path(catalog["source_results_dir"]).resolve()
        # The reviewed layout is study/results/catch22/experiments/scope/scheme.
        protected = source_results.parents[4]
        for context in contexts.values():
            layout = context["layout"]
            for path in (layout.artifact_root, layout.results_dir, layout.paper_dir, context["log"]):
                output = Path(path).resolve()
                if output == protected or protected in output.parents:
                    raise ValueError(f"Output path would modify protected reuse study: {output}")


def _copy_verified_split_caches(catalogs: list[dict], destination: Path) -> None:
    import hashlib
    for catalog in catalogs:
        source = Path(catalog["source_results_dir"]) / "metadata/split_cache"
        for path in source.glob("*.json"):
            envelope = json.loads(path.read_text())
            payload = envelope["payload"]
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            if envelope["sha256"] != hashlib.sha256(encoded).hexdigest() or payload["key"] != path.stem:
                raise ValueError(f"Protected split cache failed integrity validation: {path}")
            target = destination / path.name
            if not target.exists():
                destination.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".json.tmp")
                temporary.write_bytes(path.read_bytes())
                temporary.replace(target)


def _completed_reference(args, config: dict, context: dict, saved: dict | None = None) -> dict | None:
    """Skip only complete artifacts matching the frozen plan and current source."""
    from model_creation.run_inventory import verify_completed_configuration
    from model_creation.validation import build_validation_fingerprint
    if args.validation_scheme != "nested_cv":
        return None  # Legacy diagnostic holdout does not promise fold checkpoints.
    layout = context["layout"]
    directory = layout.results_dir / "models" / config["model_type"] / config["outcome"] / config["branch"] / config["feature_set"]
    markers = (directory / "artifacts/validation.json", directory / "artifacts/folds_manifest.json")
    if saved is None:
        saved = json.loads(context["path"].read_text()) if context["path"].exists() else {}
    if not all(path.is_file() for path in markers):
        from model_creation.run_inventory import configuration_key
        prior_configs = list(saved.get("configurations", []))
        # The in-memory request merger replaces configuration dictionaries;
        # inspect the last committed manifest before interpreting missing files
        # as merely unfinished work.
        if context["path"].exists():
            prior_configs.extend(json.loads(context["path"].read_text()).get("configurations", []))
        if any(c.get("completed_reference") for c in prior_configs
               if configuration_key(c) == configuration_key(config)):
            raise ValueError("Previously completed configuration lost required artifacts")
        return None
    split_key = "/".join(config[field] for field in ("cohort_scope", "outcome", "branch"))
    planned_hash = saved.get("split_plan", {}).get("fold_assignment_hashes", {}).get(split_key)
    if not planned_hash:
        raise ValueError("Completed run is missing its frozen fold-assignment plan")
    branch = config["branch"]
    roster = saved.get("branch_rosters", {}).get(branch, {})
    if not roster.get("dataset_sha256") or not roster.get("selected_case_hash"):
        raise ValueError("Completed run is missing its frozen modeling input/operation roster")
    dataset_path = layout.processed_dir / ("aki_features_master_wide_windowed.csv" if branch == "windowed" else "aki_features_master_wide.csv")
    record = verify_completed_configuration(directory, dataset_path=dataset_path)
    fp = record["validation_fingerprint"]
    if (fp["dataset_hash"] != roster["dataset_sha256"]
            or fp["selected_case_hash"] != roster["selected_case_hash"]):
        raise ValueError("Completed run modeling input/operation roster differs from frozen study manifest")
    resources = config["resources"]
    expected = build_validation_fingerprint(
        dataset_hash=roster["dataset_sha256"], validation_scheme=args.validation_scheme,
        model_type=config["model_type"], outcome=config["outcome"], branch=branch,
        feature_set=config["feature_set"], outer_folds=args.outer_folds,
        inner_folds=args.inner_folds, calibration_folds=args.calibration_folds, repeats=args.repeats,
        n_trials=config["n_trials"], threads_per_model=resources["threads_per_model"],
        preserve_nan=True, base_random_state=42, cohort_scope=config["cohort_scope"],
        selected_case_hash=roster["selected_case_hash"], fold_assignment_hash=planned_hash,
        diagnostic=context["manifest"]["diagnostic"],
    )
    expected["thread_environment"] = {
        "OMP_NUM_THREADS": str(resources["threads_per_model"]),
        "NUMBA_NUM_THREADS": str(resources["threads_per_model"]),
        "OPENBLAS_NUM_THREADS": str(resources["blas_threads"]),
        "MKL_NUM_THREADS": str(resources["blas_threads"]),
    }
    from model_creation.validation import compare_validation_fingerprints
    matches, reason = compare_validation_fingerprints(expected, fp)
    if not matches:
        raise ValueError(f"Completed configuration no longer matches this study: {reason}")
    return record


def _verify_holdout(directory: Path, identity: dict) -> dict:
    import pandas as pd
    from data_preparation.artifact_metadata import file_sha256
    paths = [directory / "artifacts" / name for name in
             ("validation.json", "metadata.json", "calibration.json", "threshold.json")]
    paths.extend(directory / "predictions" / name for name in ("train_oof.csv", "test.csv"))
    metadata = json.loads(paths[1].read_text())
    fp = metadata["validation_fingerprint"]
    paths.extend([directory / "preprocessor.pkl", directory / ("model.json" if fp["model_type"] == "xgboost" else "model.pkl")])
    if any(fp[k] != identity[k] for k in identity) or fp["validation_scheme"] != "holdout" or not fp["diagnostic"]:
        raise ValueError("Holdout artifacts must match the current diagnostic protocol")
    train, test = [pd.read_csv(directory / "predictions" / name) for name in ("train_oof.csv", "test.csv")]
    if (set(train.subjectid) & set(test.subjectid) or set(train.caseid) & set(test.caseid)
        or any(frame.empty or frame.caseid.duplicated().any() or not frame.y_true.isin([0, 1]).all()
               for frame in (train, test))):
        raise ValueError("Invalid patient-grouped holdout predictions")
    return {"status": "verified_diagnostic_holdout", "artifact_sha256": {str(path): file_sha256(path) for path in paths}}


def execute(args) -> int:
    import run_catch22 as launcher
    from model_creation.run_inventory import configuration_key, import_legacy_inventory
    from model_creation.split_plan import prepare_split_plan

    if args.smoke:
        args = replace_namespace(args, outer_folds=min(args.outer_folds, 2), inner_folds=min(args.inner_folds, 2),
                                 calibration_folds=min(args.calibration_folds, 2), max_workers=1,
                                 threads_per_model=min(args.threads_per_model or 2, 2), blas_threads=1)
    scopes = args.cohort_scopes or [args.cohort_scope or "main_shared"]
    scopes = list(dict.fromkeys(scopes))
    if args.rerun_completed and (args.resume or args.reuse_from):
        raise ValueError("--rerun-completed requires --no-resume and no --reuse-from, in a fresh output root")
    if not args.diagnostic and not args.smoke and args.validation_scheme == "nested_cv" and (
        args.outer_folds, args.inner_folds, args.calibration_folds, args.repeats
    ) != (5, 3, 5, 1):
        raise ValueError("Definitive validation requires 5 outer/3 inner HPO/5 calibration folds and one repeat; use --diagnostic for reduced checks.")
    contexts, configurations = {}, []
    for scope in scopes:
        scoped_args = replace_namespace(args, cohort_scope=scope)
        configs = [{**c, "cohort_scope": scope} for c in launcher._configuration_grid(scoped_args)]
        layout, log_file = launcher._prepare_primary_layout("experiment_log.txt", scope, args.validation_scheme)
        manifest = {
            "cohort_scope": scope, "validation_scheme": args.validation_scheme,
            "analysis_role": "exploratory" if scope == "mortality_waveform_sensitivity" else "sensitivity" if scope != "main_shared" else "primary",
            "diagnostic": bool(args.diagnostic or args.smoke or args.validation_scheme == "holdout"),
            "preset": args.preset, "configurations": configs,
            **{k: getattr(args, k) for k in ("outer_folds", "inner_folds", "calibration_folds", "repeats")},
            "results_dir": str(layout.results_dir), "paper_dir": str(layout.paper_dir),
            "processed_dir": str(layout.processed_dir), "raw_dir": str(layout.raw_dir),
            "scheduler": args.scheduler, "cpu_budget": args.cpu_budget or default_cpu_budget(),
            "max_workers": args.max_workers, "threads_per_model": args.threads_per_model,
            "lr_blas_threads": args.blas_threads or (1 if args.scheduler == "legacy" else 4),
            "reporting": {"bootstrap_replicates": min(args.bootstrap, 100) if args.smoke else args.bootstrap,
                          "bootstrap_seed": args.bootstrap_seed, "bootstrap_stratified": args.bootstrap_stratified},
        }
        for config in configs:
            config["resources"] = resource_plan(args, config["model_type"])
        contexts[scope] = {"layout": layout, "log": log_file, "manifest": manifest,
                           "path": layout.results_dir / "run_manifest.json"}
        configurations.extend(configs)

    catalogs = [import_legacy_inventory(Path(source), processed_dir=next(iter(contexts.values()))["layout"].processed_dir)
                for source in args.reuse_from]
    imported = {key: value for catalog in catalogs for key, value in catalog["configurations"].items()}
    if len(imported) != sum(len(c["configurations"]) for c in catalogs):
        raise ValueError("Duplicate protected configurations in reuse inventories")
    _assert_protected_paths(contexts, catalogs)
    if args.prep != "skip" and any(
        Path(catalog["source_results_dir"]).resolve().parents[4] in context["layout"].processed_dir.resolve().parents
        for catalog in catalogs for context in contexts.values()
    ):
        raise ValueError("Protected processed data require --prep skip")
    for config in configurations:
        reference = imported.get(configuration_key(config))
        if reference:
            if (args.validation_scheme != "nested_cv" or args.diagnostic or args.smoke
                or (args.outer_folds, args.inner_folds, args.calibration_folds, args.repeats) != (5, 3, 5, 1)):
                raise ValueError("Protected core reference does not match requested validation protocol")
            if reference["validation_fingerprint"]["n_trials"] != config["n_trials"]:
                raise ValueError("Protected HPO budget differs from requested configuration")
            config.update(artifact_path=reference["configuration_dir"], reuse_reference=reference)
    for context in contexts.values():
        if catalogs:
            context["manifest"]["compatibility_attestation"] = dict(catalogs[0]["compatibility_attestation"])
    if args.dry_run and not args.rerun_completed:
        for config in configurations:
            if "reuse_reference" not in config:
                reference = _completed_reference(args, config, contexts[config["cohort_scope"]])
                if reference:
                    config["completed_reference"] = reference
    summary = {**next(iter(contexts.values()))["manifest"], "cohort_scopes": scopes,
               "configurations": configurations, "protected_completed_configurations": len(imported),
               "requested_reused_configurations": sum("reuse_reference" in c for c in configurations),
               "requested_local_completed_configurations": sum("completed_reference" in c for c in configurations),
               "pending_configurations": sum("reuse_reference" not in c and "completed_reference" not in c for c in configurations),
               "scope_paths": {s: {k: str(getattr(c["layout"], k)) for k in ("results_dir", "paper_dir")}
                               for s, c in contexts.items()}}
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0

    primary = next(iter(contexts.values()))["layout"]
    control = primary.artifact_root / "control"
    launcher.enforce_storage_policy({"control_dir": control, "processed_dir": primary.processed_dir,
                                     "results_dir": primary.results_dir, "paper_dir": primary.paper_dir})
    if any(primary.artifact_root.resolve() == Path(source).resolve() for source in args.reuse_from):
        raise ValueError("Reuse source must be a different read-only study")
    with launcher._exclusive_run(control / ".controller.lock"):
        control.mkdir(parents=True, exist_ok=True)
        if args.rerun_completed and any(c["path"].exists() for c in contexts.values()):
            raise ValueError("Explicit reruns require a fresh output root")
        if catalogs:
            launcher._write_manifest(control / "reused_inventory.json", {"sources": catalogs})
            _copy_verified_split_caches(catalogs, control / "split_cache")
        launcher._write_manifest(control / "execution_plan.json", summary)
        identity = launcher._study_identity()
        for context in contexts.values():
            if catalogs:
                context["manifest"]["compatibility_attestation"]["target_study_identity"] = identity
        jobs, lookup, failed_scopes = [], {}, set()
        for scope, context in contexts.items():
            layout, manifest = context["layout"], context["manifest"]
            launcher._ensure_paths({"results_dir": layout.results_dir, "paper_dir": layout.paper_dir,
                                    "log_file": context["log"]})
            if args.refresh_links:
                launcher.refresh_repo_convenience_paths(launcher.PROJECT_ROOT, processed_dir=layout.processed_dir,
                                                        results_dir=layout.results_dir, paper_dir=layout.paper_dir)
            env = launcher._base_env(layout, log_file=context["log"])
            env.update(COHORT_SCOPE=scope, VALIDATION_SCHEME=args.validation_scheme,
                       AKI_SPLIT_CACHE_DIR=str(control / "split_cache"), AKI_REPORT_CPU_BUDGET=str(summary["cpu_budget"]))
            context["env"] = env
            with launcher._open_log_file(context["log"]) as log:
                if args.report_only:
                    if not context["path"].exists():
                        raise ValueError("--report-only requires an existing run_manifest.json")
                    saved = json.loads(context["path"].read_text())
                    saved["reporting"] = manifest["reporting"]
                    context["manifest"] = manifest = saved
                else:
                    manifest["study_identity"] = identity
                    # Read-only validation does not create locks in the processed source.
                    with launcher._exclusive_run(control / ".preparation.lock"):
                        if launcher._ensure_data_prepared(args.prep, env=env, processed_dir=layout.processed_dir, log_handle=log):
                            return 1
                    manifest["branch_rosters"] = launcher._freeze_branch_rosters(
                        layout.processed_dir, layout.results_dir, scope, [c["branch"] for c in manifest["configurations"]])
                    if context["path"].exists():
                        previous = json.loads(context["path"].read_text())
                        for field in ("study_identity", "validation_scheme", "diagnostic", "outer_folds", "inner_folds", "calibration_folds", "repeats", "branch_rosters"):
                            if field == "branch_rosters":
                                for branch, roster in previous.get(field, {}).items():
                                    if branch in manifest[field] and roster != manifest[field][branch]:
                                        raise ValueError("Frozen operation roster or source changed")
                                    manifest[field].setdefault(branch, roster)
                            elif previous.get(field) != manifest[field]:
                                raise ValueError(f"Existing study changed {field}; use a fresh output root")
                        current = {configuration_key(c): c for c in previous["configurations"]}
                        for c in manifest["configurations"]:
                            old = current.get(configuration_key(c))
                            if old and (old["n_trials"] != c["n_trials"] or old.get("resources") != c.get("resources")):
                                raise ValueError("HPO or numerical execution settings changed within the study")
                            current[configuration_key(c)] = c
                        requested = manifest["configurations"]
                        manifest["configurations"] = list(current.values())
                        manifest["execution_history"] = previous.get("execution_history", [])
                        if "split_plan" in previous:
                            manifest["split_plan"] = previous["split_plan"]
                    else:
                        requested = manifest["configurations"]
                    pending = []
                    for config in requested:
                        if "reuse_reference" in config:
                            continue
                        directory = layout.results_dir / "models" / config["model_type"] / config["outcome"] / config["branch"] / config["feature_set"]
                        if not args.rerun_completed:
                            reference = _completed_reference(args, config, context, manifest)
                        else:
                            reference = None
                        if reference:
                            config["completed_reference"] = reference
                            continue
                        pending.append(config)
                    try:
                        if pending and args.validation_scheme == "nested_cv":
                            old_hashes = manifest.get("split_plan", {}).get("fold_assignment_hashes", {})
                            plan = prepare_split_plan(
                                layout.processed_dir, pending, control / "split_cache", outer_folds=args.outer_folds,
                                inner_folds=args.inner_folds, calibration_folds=args.calibration_folds,
                                workers=min(8, summary["cpu_budget"]))
                            plan["fold_assignment_hashes"] = {**old_hashes, **plan.get("fold_assignment_hashes", {})}
                            manifest["split_plan"] = plan
                    except ValueError as exc:
                        manifest.update(status="infeasible", failures=[str(exc)])
                        failed_scopes.add(scope)
                        pending = []
                    for config in pending:
                        resources = config["resources"]
                        fit_env = dict(env)
                        fit_env.update(OMP_NUM_THREADS=str(resources["threads_per_model"]), NUMBA_NUM_THREADS=str(resources["threads_per_model"]),
                                       OPENBLAS_NUM_THREADS=str(resources["blas_threads"]), MKL_NUM_THREADS=str(resources["blas_threads"]),
                                       AKI_CONFIG_CPU_RESERVATION=str(resources["cpu_reservation"]), AKI_TRIAL_HISTORY_DIR=str(
                                           layout.results_dir / "models" / config["model_type"] / config["outcome"] / config["branch"] / config["feature_set"] / "artifacts/trials"))
                        if args.validation_scheme == "nested_cv":
                            split_key = "/".join(config[k] for k in ("cohort_scope", "outcome", "branch"))
                            fit_env["AKI_EXPECTED_FOLD_HASH"] = manifest["split_plan"]["fold_assignment_hashes"].get(split_key, "")
                        cli = ["--outcome", config["outcome"], "--branch", config["branch"], "--feature_set", config["feature_set"],
                               "--model_type", config["model_type"], "--cohort-scope", scope, "--validation-scheme", args.validation_scheme]
                        for name in ("outer_folds", "inner_folds", "calibration_folds", "repeats"):
                            cli.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
                        for name in ("max_workers", "threads_per_model"):
                            cli.extend(["--" + name.replace("_", "-"), str(resources[name])])
                        cli.extend(["--n-trials", str(config["n_trials"])])
                        for flag, enabled in (("--resume", args.resume), ("--save-final-refit", args.save_final_refit),
                                              ("--diagnostic", manifest["diagnostic"]), ("--smoke_test", args.smoke)):
                            if enabled:
                                cli.append(flag)
                        key = configuration_key(config)
                        jobs.append(ConfigJob(key, config["model_type"], [sys.executable, "-m", "model_creation.step_07_train_evaluate", *cli],
                                              fit_env, launcher.PROJECT_ROOT, control / "logs" / (key.replace("/", "__") + ".log"),
                                              resources["cpu_reservation"], priority=(config["branch"] != "windowed", -len(config["feature_set"]))))
                        lookup[key] = (context, config)
                manifest.setdefault("status", "running")
                launcher._write_manifest(context["path"], manifest)

        def finished(record):
            context, config = lookup[record["key"]]
            manifest = context["manifest"]
            if record["exit_code"] == 0:
                directory = context["layout"].results_dir / "models" / config["model_type"] / config["outcome"] / config["branch"] / config["feature_set"]
                try:
                    config["completed_reference"] = (_verify_holdout(directory, identity) if args.validation_scheme == "holdout"
                        else _completed_reference(args, config, context, manifest))
                    if config["completed_reference"] is None:
                        raise ValueError(f"Successful child did not save complete artifacts: {directory}")
                except (ValueError, OSError) as exc:
                    record.update(exit_code=1, status="artifact_validation_failed", error=str(exc))
            manifest.setdefault("execution_history", []).append({**config, **record, "finished_at": datetime.now(timezone.utc).isoformat()})
            if record["exit_code"] != 0:
                failed_scopes.add(config["cohort_scope"])
            launcher._write_manifest(context["path"], manifest)

        if launcher._study_identity() != identity:
            raise ValueError("Source or environment changed during preflight; freeze the checkout before running")
        def started(record):
            if launcher._study_identity() != identity:
                raise ValueError("Frozen source/environment changed while scheduling configurations")
        if jobs:
            run_configurations(jobs, cpu_budget=summary["cpu_budget"], max_concurrent_configs=args.max_concurrent_configs,
                               profile=args.profile_resources, output_dir=control / "scheduler", legacy=args.scheduler == "legacy",
                               on_started=started, on_finished=finished)
        for scope, context in contexts.items():
            manifest = context["manifest"]
            if scope in failed_scopes:
                if manifest["status"] != "infeasible":
                    manifest["status"] = "failed"
                print(f"{scope}: validation failed; skipping metrics/report generation")
            else:
                manifest["status"] = "models_complete"
                launcher._write_manifest(context["path"], manifest)
                if not args.skip_report:
                    with launcher._open_log_file(context["log"]) as log:
                        result = launcher._run_reporting(context["layout"].results_dir, args.parallel_backend,
                            env=context["env"], log_handle=log, bootstrap=manifest["reporting"]["bootstrap_replicates"],
                            bootstrap_seed=args.bootstrap_seed, bootstrap_stratified=args.bootstrap_stratified)
                    manifest["status"] = "validated" if result == 0 else "reporting_failed"
                    if result:
                        failed_scopes.add(scope)
            launcher._write_manifest(context["path"], manifest)
        return int(bool(failed_scopes))


def replace_namespace(args, **changes):
    import argparse
    return argparse.Namespace(**{**vars(args), **changes})
