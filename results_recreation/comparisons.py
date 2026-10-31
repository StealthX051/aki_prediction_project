"""Prespecified, paired comparisons of locked outer-test predictions.

Metric replicates are reused, not recomputed for each contrast. Decision curves
reproduce the exact patient draws identified in the metric summary. No fitting
or model selection is performed here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from results_recreation import metrics_summary as metrics
from results_recreation.delong import gated_delong, add_holm_adjustment
from reporting.decision_curve import (
    DEFAULT_RANGES, outcome_thresholds, threshold_grid_metadata, cluster_resample_counts, bootstrap_net_benefit,
)

CONTEXT = ("cohort_scope", "validation_scheme", "outcome", "branch")
FAMILIES = ("logistic_regression", "xgboost", "ebm")
CORE = ("preop_only", "all_waveforms", "preop_and_all_waveforms")
METRICS = ("auprc", "auroc", "brier", "sensitivity", "specificity", "precision", "f1", "accuracy")
ARTIFACTS = {"paired": "paired_comparisons.csv", "dca": "decision_curve_summary.csv",
             "dca_comparisons": "decision_curve_comparisons.csv", "delong": "delong_comparisons.csv"}
PAIR_COLUMNS = [*CONTEXT, "comparison_id", "comparison_type", "designation", "reference_model",
                "reference_feature_set", "candidate_model", "candidate_feature_set"]
PROVENANCE_COLUMNS = ["bootstrap_draw_hash", "reference_prediction_hash", "candidate_prediction_hash"]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=lambda v: v.item() if isinstance(v, np.generic) else str(v)).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calculation_fingerprint():
    root = Path(__file__).resolve().parents[1]
    paths = ("results_recreation/comparisons.py", "results_recreation/metrics_summary.py",
             "results_recreation/delong.py", "reporting/decision_curve.py")
    import scipy
    from sklearn import __version__ as sklearn_version
    return {"uncertainty_protocol_version": metrics.UNCERTAINTY_PROTOCOL_VERSION,
            "code": {p: _file_hash(root / p) for p in paths},
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn_version,
            "dca_grids": {key: threshold_grid_metadata(outcome_thresholds(key)) for key in DEFAULT_RANGES}}


def manifest_identity(manifest):
    if manifest is None:
        return None
    # Runtime status/timings change after reporting; scientific selection does not.
    fields = (*CONTEXT[:2], "diagnostic", "branch_rosters", "study_identity", "outer_folds",
              "inner_folds", "calibration_folds", "repeats", "reporting")
    return {**{k: manifest.get(k) for k in fields},
            "configurations": sorted(manifest.get("configurations", []), key=lambda c: json.dumps(c, sort_keys=True))}


def _key(ps):
    return (*[getattr(ps, c) for c in CONTEXT], metrics._model_id(ps.model_name), ps.feature_set)


def _canonical_sets(prediction_sets):
    result = {}
    for ps in prediction_sets:
        ps = replace(ps, df=ps.df.sort_values("caseid", kind="stable").reset_index(drop=True))
        if _key(ps) in result:
            raise metrics.ValidationError(f"Duplicate comparison configuration: {_key(ps)}")
        if ps.df.caseid.isna().any() or ps.df.caseid.duplicated().any():
            raise metrics.ValidationError("Comparisons require unique nonmissing operation IDs")
        if not ps.df.y_true.isin([0, 1]).all():
            raise metrics.ValidationError("Comparisons require binary outcomes")
        risks = ps.df.y_prob_calibrated.to_numpy(dtype=float)
        if not np.isfinite(risks).all() or ((risks < 0) | (risks > 1)).any():
            raise metrics.ValidationError("Comparisons require finite calibrated probabilities in [0,1]")
        result[_key(ps)] = ps
    return result


def _inventory(context, available):
    entries = []
    def add(reference, candidate, kind):
        primary = (context[0] == "main_shared" and context[3] == "non_windowed"
                   and context[2] in {"any_aki", "icu_admission"} and kind == "input"
                   and reference[1] == CORE[0] and candidate[1] == CORE[2])
        designation = ("primary" if primary else "secondary"
                       if context[0] == "main_shared" and context[3] == "non_windowed" and kind != "ablation"
                       else "exploratory")
        identity = {**dict(zip(CONTEXT, context)), "comparison_type": kind,
                    "designation": designation, "reference_model": reference[0],
                    "reference_feature_set": reference[1], "candidate_model": candidate[0],
                    "candidate_feature_set": candidate[1]}
        entries.append({**identity, "comparison_id": _digest(identity)[:20],
                        "availability": "available" if reference in available and candidate in available else "not_requested"})
    for family in FAMILIES:
        for ref, cand in ((CORE[0], CORE[2]), (CORE[0], CORE[1]), (CORE[1], CORE[2])):
            add((family, ref), (family, cand), "input")
    for feature in CORE:
        for ref, cand in ((FAMILIES[0], FAMILIES[1]), (FAMILIES[0], FAMILIES[2]), (FAMILIES[1], FAMILIES[2])):
            add((ref, feature), (cand, feature), "family")
    for family, feature in sorted(available):
        if family in FAMILIES and feature not in CORE:
            add((family, CORE[0]), (family, feature), "ablation")
    return entries


def _summary_row(summary, ps):
    mask = pd.Series(True, index=summary.index)
    for field in CONTEXT:
        mask &= summary[field].eq(getattr(ps, field))
    mask &= summary.model_name.map(metrics._model_id).eq(metrics._model_id(ps.model_name))
    mask &= summary.feature_set.eq(ps.feature_set)
    rows = summary.loc[mask]
    if len(rows) != 1:
        raise metrics.ValidationError("Comparison metrics require exactly one summary row per configuration")
    return rows.iloc[0]


def _replicates(bootstrap, ps, row, n_bootstrap, expected_draw_hash):
    if n_bootstrap == 0:
        return pd.DataFrame(columns=METRICS)
    if bootstrap is None:
        raise metrics.ValidationError("Comparison uncertainty requires saved metric replicates")
    mask = pd.Series(True, index=bootstrap.index)
    for field in CONTEXT:
        mask &= bootstrap[field].eq(getattr(ps, field))
    mask &= bootstrap.model_name.map(metrics._model_id).eq(metrics._model_id(ps.model_name))
    mask &= bootstrap.feature_set.eq(ps.feature_set)
    selected = bootstrap.loc[mask].sort_values("bootstrap_id")
    if not np.array_equal(selected.bootstrap_id.to_numpy(), np.arange(n_bootstrap)):
        raise metrics.ValidationError("Metric bootstrap replicates have missing or duplicated draw IDs")
    for column in ("prediction_content_hash", "uncertainty_protocol_version", "bootstrap_replicates",
                   "bootstrap_seed", "bootstrap_stratified"):
        if column not in selected or not selected[column].eq(row[column]).all():
            raise metrics.ValidationError(f"Stale bootstrap replicate provenance: {column}")
    if not selected.bootstrap_draw_hash.eq(expected_draw_hash).all():
        raise metrics.ValidationError("Metric and DCA bootstrap draws differ")
    return selected.set_index("bootstrap_id")


def build_comparison_artifacts(prediction_sets, summary, bootstrap, *, n_bootstrap=5000,
                               stratified_bootstrap=False, bootstrap_seed=0,
                               run_manifest=None, strict=True):
    sets = _canonical_sets(prediction_sets)
    if not sets:
        raise metrics.ValidationError("No predictions for comparison")
    if n_bootstrap < 0:
        raise metrics.ValidationError("Bootstrap replicate count must be nonnegative")
    policy = {"bootstrap_replicates": int(n_bootstrap), "bootstrap_seed": int(bootstrap_seed),
              "bootstrap_stratified": bool(stratified_bootstrap)}
    if run_manifest is not None and run_manifest.get("reporting") is not None:
        if any(run_manifest["reporting"].get(k) != v for k, v in policy.items()):
            raise metrics.ValidationError("Requested bootstrap policy differs from run manifest; record it with launcher --report-only")
    if strict and run_manifest is not None:
        expected = {(run_manifest["cohort_scope"], run_manifest["validation_scheme"], c["outcome"],
                     c["branch"], c["model_type"], c["feature_set"]) for c in run_manifest["configurations"]}
        if expected - sets.keys():
            raise metrics.ValidationError("Comparison inventory is missing manifest-promised configurations")
    populations = {}
    for key, ps in sets.items():
        populations.setdefault(key[:4], {})[key[4:]] = ps
    paired_rows, dca_tables, dca_pairs, delong_rows, entries = [], [], [], [], []
    for context, available in sorted(populations.items()):
        first = next(iter(available.values()))
        for ps in available.values():
            patient_column = metrics._get_cluster_column(ps.df)
            if patient_column == "caseid" or ps.df[patient_column].isna().any():
                raise metrics.ValidationError("Patient IDs are required for comparison uncertainty")
            if strict and ps.validation_scheme == "nested_cv" and not {"outer_fold_id", "repeat_id"}.issubset(ps.df):
                raise metrics.ValidationError("Nested comparison predictions require outer fold and repeat identities")
            if not metrics._prediction_frames_are_aligned(first.df, ps.df):
                raise metrics.ValidationError("Comparison case/patient/truth/fold rosters differ")
            if ps.pipeline != first.pipeline or ps.analysis_status != first.analysis_status:
                raise metrics.ValidationError("Comparison pipelines or analysis statuses differ")
        cluster_col = metrics._get_cluster_column(first.df)
        groups = first.df[cluster_col].to_numpy()
        indices = metrics._generate_bootstrap_indices(first.df, n_bootstrap, stratified_bootstrap, bootstrap_seed)
        draw_hash = metrics.bootstrap_draw_hash(indices) if n_bootstrap else "none"
        cluster_counts = cluster_resample_counts(groups, indices)
        del indices
        context_entries = _inventory(context, available)
        entries.extend(context_entries)
        draws, point, samples, curve_points = {}, {}, {}, {}
        thresholds = outcome_thresholds(context[2])
        for key, ps in available.items():
            row = _summary_row(summary, ps)
            if (row.uncertainty_protocol_version != metrics.UNCERTAINTY_PROTOCOL_VERSION
                    or row.prediction_content_hash != metrics._prediction_content_hash(ps)
                    or row.bootstrap_replicates != n_bootstrap or row.bootstrap_seed != bootstrap_seed
                    or bool(row.bootstrap_stratified) != stratified_bootstrap
                    or row.bootstrap_draw_hash != draw_hash):
                raise metrics.ValidationError("Comparison summary and resampling provenance differ")
            point[key] = row
            samples[key] = _replicates(bootstrap, ps, row, n_bootstrap, draw_hash)
            table, draws[key] = bootstrap_net_benefit(ps.df.y_true.to_numpy(), ps.df.y_prob_calibrated.to_numpy(),
                                                    thresholds, groups=groups, cluster_counts=cluster_counts)
            curve_points[key] = table.model.to_numpy()
            for name, value in {**dict(zip(CONTEXT, context)), "model_name": ps.model_name,
                                "model_type": key[0], "feature_set": key[1], "analysis_status": ps.analysis_status,
                                "n_operations": len(ps.df), "n_patients": pd.Series(groups).nunique(),
                                "bootstrap_draw_hash": draw_hash,
                                "prediction_content_hash": row.prediction_content_hash}.items():
                table[name] = value
            dca_tables.append(table)
        for entry in context_entries:
            if entry["availability"] != "available":
                continue
            ref_key = (entry["reference_model"], entry["reference_feature_set"])
            cand_key = (entry["candidate_model"], entry["candidate_feature_set"])
            identity = {k: entry[k] for k in PAIR_COLUMNS}
            identity.update(n_operations=len(first.df), n_patients=pd.Series(groups).nunique(),
                            analysis_status=first.analysis_status, bootstrap_draw_hash=draw_hash,
                            reference_prediction_hash=point[ref_key].prediction_content_hash,
                            candidate_prediction_hash=point[cand_key].prediction_content_hash)
            for metric in METRICS:
                differences = (samples[cand_key][metric] - samples[ref_key][metric]).to_numpy() if n_bootstrap else []
                lower, upper, count = metrics._bootstrap_interval(
                    differences, n_bootstrap, label=f"{entry['comparison_id']}/{metric}", warn_missing=n_bootstrap > 0)
                designation = entry["designation"]
                if designation == "primary" and metric != "auprc":
                    designation = "secondary"
                paired_rows.append({**identity, "designation": designation, "metric": metric,
                                    "reference_estimate": point[ref_key][metric],
                                    "candidate_estimate": point[cand_key][metric],
                                    "delta": point[cand_key][metric] - point[ref_key][metric],
                                    "lower": lower, "upper": upper, "bootstrap_valid": count,
                                    "interval_status": "estimated" if count >= 2 else "not_estimated"})
            differences = draws[cand_key] - draws[ref_key]
            if n_bootstrap >= 2:
                lower, upper = np.percentile(differences, [2.5, 97.5], axis=0)
            else:
                lower = upper = np.full(len(thresholds), np.nan)
            # Point estimates use the original operations, not bootstrap means.
            delta = curve_points[cand_key] - curve_points[ref_key]
            dca_pairs.extend({**identity, "designation": "exploratory", "threshold_probability": t, "delta": d, "lower": lo,
                              "upper": hi, "bootstrap_valid": n_bootstrap}
                             for t, d, lo, hi in zip(thresholds, delta, lower, upper))
            delong_rows.append({**identity, "designation": "secondary" if entry["designation"] == "primary"
                                else entry["designation"], **gated_delong(available[ref_key], available[cand_key])})
    paired = pd.DataFrame(paired_rows, columns=[*PAIR_COLUMNS, *PROVENANCE_COLUMNS, "n_operations", "n_patients", "analysis_status",
                         "metric", "reference_estimate", "candidate_estimate", "delta", "lower", "upper",
                         "bootstrap_valid", "interval_status"])
    dca_pairs = pd.DataFrame(dca_pairs, columns=[*PAIR_COLUMNS, *PROVENANCE_COLUMNS, "n_operations", "n_patients", "analysis_status",
                            "threshold_probability", "delta", "lower", "upper", "bootstrap_valid"])
    delong = pd.DataFrame(delong_rows, columns=[*PAIR_COLUMNS, *PROVENANCE_COLUMNS, "n_operations", "n_patients", "analysis_status",
                         "reference_auc", "candidate_auc", "delta", "variance", "p_value", "log_p_value",
                         "n_positive", "n_negative", "method", "status", "reason"])
    delong = add_holm_adjustment(delong)
    inventory = {"comparison_status": "complete" if all(e["availability"] == "available" for e in entries) else "partial",
                 "comparisons": entries, "available_comparisons": sum(e["availability"] == "available" for e in entries),
                 "requested_core_comparisons": sum(e["comparison_type"] != "ablation" for e in entries),
                 "bootstrap_replicates": int(n_bootstrap), "bootstrap_seed": int(bootstrap_seed),
                 "bootstrap_stratified": bool(stratified_bootstrap), "confidence_level": .95,
                 "uncertainty_status": "estimated" if n_bootstrap >= 2 else "not_estimated",
                 "uncertainty_scope": "nominal pointwise patient-cluster percentile intervals conditional on saved fitted models",
                 "estimand": "operation-level; all operations retained within sampled patient clusters",
                 "calculation_fingerprint": calculation_fingerprint(),
                 "manifest_identity": manifest_identity(run_manifest),
                 "prediction_hashes": {_digest(k): metrics._prediction_content_hash(ps) for k, ps in sets.items()},
                 "dca_ranges": {outcome: outcome_thresholds(outcome).tolist() for outcome in sorted({k[2] for k in sets})}}
    result = {"paired": paired, "dca": pd.concat(dca_tables, ignore_index=True), "dca_comparisons": dca_pairs,
              "delong": delong, "inventory": inventory}
    for frame in (result[key] for key in ARTIFACTS):
        for column in ("bootstrap_replicates", "bootstrap_seed", "bootstrap_stratified", "confidence_level"):
            frame[column] = inventory[column]
        frame["uncertainty_protocol_version"] = metrics.UNCERTAINTY_PROTOCOL_VERSION
        frame["uncertainty_scope"] = inventory["uncertainty_scope"]
    return result


def write_comparison_artifacts(artifacts, tables_dir, *, paper_tables_dir=None):
    tables_dir = Path(tables_dir)
    from artifact_paths import enforce_storage_policy
    enforce_storage_policy({"comparison_tables": tables_dir,
                            **({"paper_comparison_tables": paper_tables_dir} if paper_tables_dir else {})})
    tables_dir.mkdir(parents=True, exist_ok=True)
    inventory = dict(artifacts["inventory"])
    inventory["artifact_hashes"] = {}
    for key, name in ARTIFACTS.items():
        path = tables_dir / name
        temporary = path.with_suffix(".csv.tmp")
        artifacts[key].to_csv(temporary, index=False)
        temporary.replace(path)
        inventory["artifact_hashes"][name] = _file_hash(path)
    inventory["metrics_summary_sha256"] = _file_hash(tables_dir / "metrics_summary.csv")
    path = tables_dir / "comparison_inventory.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    if paper_tables_dir is not None:
        import shutil
        Path(paper_tables_dir).mkdir(parents=True, exist_ok=True)
        for name in (*ARTIFACTS.values(), path.name):
            shutil.copy2(tables_dir / name, Path(paper_tables_dir) / name)


def load_comparison_artifacts(tables_dir, prediction_sets, summary_df, *, run_manifest=None):
    tables_dir = Path(tables_dir)
    path = tables_dir / "comparison_inventory.json"
    if not path.exists():
        raise metrics.ValidationError("Missing comparison artifacts; regenerate metrics_summary")
    inventory = json.loads(path.read_text())
    if inventory.get("calculation_fingerprint") != calculation_fingerprint():
        raise metrics.ValidationError("Stale comparison calculation protocol; regenerate metrics_summary")
    if isinstance(run_manifest, (str, Path)):
        run_manifest = json.loads(Path(run_manifest).read_text())
    if inventory.get("manifest_identity") != manifest_identity(run_manifest):
        raise metrics.ValidationError("Stale comparison run selection; regenerate metrics_summary")
    sets = _canonical_sets(prediction_sets)
    if inventory.get("prediction_hashes") != {_digest(k): metrics._prediction_content_hash(ps) for k, ps in sets.items()}:
        raise metrics.ValidationError("Stale comparison prediction provenance; regenerate metrics_summary")
    if inventory.get("metrics_summary_sha256") != _file_hash(tables_dir / "metrics_summary.csv"):
        raise metrics.ValidationError("Stale comparison metric summary; regenerate metrics_summary")
    for column in ("bootstrap_replicates", "bootstrap_seed", "bootstrap_stratified"):
        if column not in summary_df or not summary_df[column].eq(inventory[column]).all():
            raise metrics.ValidationError("Comparison and metric resampling policies differ")
    artifacts = {"inventory": inventory}
    for key, name in ARTIFACTS.items():
        artifact = tables_dir / name
        if not artifact.exists() or inventory.get("artifact_hashes", {}).get(name) != _file_hash(artifact):
            raise metrics.ValidationError(f"Missing or altered comparison artifact: {name}")
        artifacts[key] = pd.read_csv(artifact)
    return artifacts
