"""Generate tables and figures from standardized reporting artifacts.

This script consumes the scoped experiments root's `tables/metrics_summary.csv`,
verified comparison tables, and saved `models/**/predictions/test.csv` files.
It regenerates paper tables and figures without fitting models, recalibration,
threshold selection, or new bootstrap calculations. `--legacy` explicitly
permits historical artifacts and point-only decision curves.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np

from artifact_paths import enforce_storage_policy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results_recreation.metrics_summary import (
    PredictionSet, UNCERTAINTY_PROTOCOL_VERSION, crawl_predictions, compute_point_metrics,
    _model_id, _prediction_content_hash,
)
from results_recreation.results_analysis import (
    FIGURES_DIR,
    PlotConfig,
    RESULTS_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    generate_docx_report,
    generate_html_tables,
    generate_markdown_report,
    generate_pdf_report,
    plot_curves,
    prepare_metrics_for_display,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _load_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load the consolidated metrics summary CSV."""

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics summary not found at {metrics_path}. Run results_recreation.metrics_summary first."
        )

    return pd.read_csv(metrics_path)


def _analysis_role(manifest: dict) -> str:
    defaults = {"main_shared": "primary", "icu_waveform_sensitivity": "sensitivity",
                "mortality_waveform_sensitivity": "exploratory"}
    scope = manifest.get("cohort_scope")
    role = manifest.get("analysis_role", defaults.get(scope, "primary"))
    if scope == "mortality_waveform_sensitivity" and role != "exploratory":
        raise ValueError("Mortality sensitivity reports must retain their exploratory scientific role")
    return role


def _report_context_notes(manifest: dict) -> list[str]:
    """Carry scientific role and cohort denominators into every human export."""
    if not manifest:
        return []
    role = _analysis_role(manifest)
    notes = [f"Scientific analysis role: {role}. Validation status records validation completeness."]
    for branch, cohort in manifest.get("branch_rosters", {}).items():
        if not {"operations", "patients"}.issubset(cohort):
            continue
        note = f"{branch}: {cohort['operations']} operations; {cohort['patients']} patients."
        mortality = cohort.get("outcomes", {}).get("mortality")
        if mortality:
            note += f" Mortality-positive operations: {mortality['events']}; positive patients: {mortality['event_patients']}."
        notes.append(note)
    return notes


def _prediction_sets_to_frame(prediction_sets: Iterable[PredictionSet]) -> pd.DataFrame:
    """Combine validated prediction sets into a single DataFrame for plotting."""

    frames = []
    for pred_set in prediction_sets:
        df = pred_set.df.copy()
        df["outcome"] = pred_set.outcome
        df["branch"] = pred_set.branch
        df["feature_set"] = pred_set.feature_set
        df["model_name"] = pred_set.model_name
        df["cohort_scope"] = pred_set.cohort_scope
        df["validation_scheme"] = pred_set.validation_scheme
        df["analysis_status"] = pred_set.analysis_status

        # Ensure calibrated column exists for plotting fallbacks
        if "y_prob_calibrated" not in df.columns and "y_prob_raw" in df.columns:
            df["y_prob_calibrated"] = df["y_prob_raw"]

        frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_comparisons(tables_dir, prediction_sets, summary, *, run_manifest=None):
    from results_recreation.comparisons import load_comparison_artifacts
    return load_comparison_artifacts(tables_dir, prediction_sets, summary, run_manifest=run_manifest)


def _export_comparison_tables(artifacts: dict, tables_dir: Path, *, report_notes=()) -> None:
    """Present the prespecified contrasts without altering their saved statistics."""
    from reporting.display_dictionary import load_display_dictionary
    from reporting.feature_set_display import feature_set_sort_key
    from reporting.manuscript_assets import export_dataframe_bundle

    paired = artifacts["paired"]
    if paired.empty:
        return
    labels = load_display_dictionary().feature_set_labels()
    family_labels = {"logistic_regression": "Logistic regression", "xgboost": "XGBoost", "ebm": "EBM"}
    metric_labels = {"auprc": "AP", "auroc": "AUROC", "brier": "Brier score"}
    family_order = {name: index for index, name in enumerate(family_labels)}
    metric_order = {name: index for index, name in enumerate(metric_labels)}
    status = artifacts["inventory"]["comparison_status"]
    notes = (
        "Differences are candidate minus reference. Higher AP/AUROC and lower Brier scores are better. "
        "Intervals are nominal 95% paired patient-cluster percentile intervals conditional on saved fitted-model "
        "predictions; they do not establish simultaneous superiority. Shown estimates are never used to select "
        "thresholds or refit models. Comparison coverage: " + status + "."
    )
    notes = "\n\n".join([*report_notes, notes])
    sections = [
        ("comparison_primary", "Primary waveform added-value comparisons",
         paired.loc[paired.designation.eq("primary") & paired.metric.eq("auprc")]),
        ("comparison_families", "Secondary matched-family comparisons",
         paired.loc[paired.comparison_type.eq("family") & paired.branch.eq("non_windowed")
                    & paired.metric.isin(metric_labels)]),
    ]
    def number(value):
        return f"{value:.3f}" if pd.notna(value) and np.isfinite(value) else "NA"
    for name, title, selected in sections:
        if selected.empty:
            continue
        if name == "comparison_families" and selected.designation.eq("exploratory").all():
            title = "Exploratory matched-family sensitivity comparisons"
        selected = selected.loc[sorted(selected.index, key=lambda idx: (
            selected.loc[idx, "outcome"],
            feature_set_sort_key(selected.loc[idx, "candidate_feature_set"]),
            family_order.get(selected.loc[idx, "candidate_model"], len(family_order)),
            family_order.get(selected.loc[idx, "reference_model"], len(family_order)),
            metric_order.get(selected.loc[idx, "metric"], len(metric_order)),
        ))].copy()
        display_rows = []
        for row in selected.itertuples(index=False):
            display_rows.append({
                "Outcome": {"any_aki": "AKI", "icu_admission": "ICU admission", "mortality": "In-hospital mortality"}.get(row.outcome, row.outcome),
                "Reference": f"{family_labels.get(row.reference_model, row.reference_model)}: {labels.get(row.reference_feature_set, row.reference_feature_set)}",
                "Candidate": f"{family_labels.get(row.candidate_model, row.candidate_model)}: {labels.get(row.candidate_feature_set, row.candidate_feature_set)}",
                "Metric": metric_labels.get(row.metric, row.metric),
                "Difference (95% interval)": f"{number(row.delta)} ({number(row.lower)}, {number(row.upper)})",
                "Valid resamples": int(row.bootstrap_valid),
                "Operations / patients": f"{int(row.n_operations)} / {int(row.n_patients)}",
            })
        export_dataframe_bundle(
            tables_dir / name, title=f"{title} [{status}] [{', '.join(sorted(selected.analysis_status.unique()))}]", dataframe=selected,
            display_dataframe=pd.DataFrame(display_rows), notes=notes,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(strict=os.getenv("REPORT_STRICT", "1") == "1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict", dest="strict", action="store_true")
    mode.add_argument("--legacy", dest="strict", action="store_false")
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--explain-folds", action="store_true")
    args = parser.parse_args()
    prediction_sets = crawl_predictions(RESULTS_DIR, strict=args.strict, run_manifest=args.run_manifest, require_reference=args.strict)
    metrics_path = TABLES_DIR / "metrics_summary.csv"
    summary = _load_metrics(metrics_path)
    comparisons = None
    if args.strict:
        selected_rows = []
        for ps in prediction_sets:
            mask = (summary.outcome.eq(ps.outcome) & summary.branch.eq(ps.branch)
                    & summary.feature_set.eq(ps.feature_set)
                    & summary.model_name.map(_model_id).eq(_model_id(ps.model_name)))
            if "cohort_scope" not in summary or "validation_scheme" not in summary:
                raise ValueError("Strict report requires scoped metric summary metadata")
            mask &= summary.cohort_scope.eq(ps.cohort_scope) & summary.validation_scheme.eq(ps.validation_scheme)
            rows = summary.loc[mask]
            if len(rows) != 1:
                raise ValueError("Metric summary does not contain exactly one requested prediction configuration")
            if "analysis_status" not in rows or rows.iloc[0].analysis_status != ps.analysis_status:
                raise ValueError("Metric summary has stale diagnostic/definitive status")
            if rows.iloc[0].get("uncertainty_protocol_version") != UNCERTAINTY_PROTOCOL_VERSION:
                raise ValueError("Metric summary uses a stale uncertainty protocol; regenerate metrics_summary")
            if ("prediction_content_hash" not in rows
                    or rows.iloc[0].prediction_content_hash != _prediction_content_hash(ps)):
                raise ValueError(f"Stale metric summary prediction/protocol provenance for {ps.path}; regenerate metrics_summary")
            row = rows.iloc[0]
            if row.get("delta_mode") == "reference":
                references = [reference for reference in prediction_sets
                    if (reference.cohort_scope, reference.validation_scheme, reference.outcome, reference.branch,
                        reference.pipeline, _model_id(reference.model_name), reference.feature_set) ==
                       (ps.cohort_scope, ps.validation_scheme, ps.outcome, ps.branch, ps.pipeline,
                        _model_id(ps.model_name), row.get("reference_feature_set"))]
                if (_model_id(row.get("reference_model_name")) != _model_id(ps.model_name)
                        or len(references) != 1
                        or row.get("reference_prediction_content_hash") != _prediction_content_hash(references[0])):
                    raise ValueError(f"Stale metric summary reference provenance for {ps.path}; regenerate metrics_summary")
            selected_rows.extend(rows.index)
            for metric, value in compute_point_metrics(ps).items():
                if metric not in rows or not np.isclose(rows.iloc[0][metric], value, equal_nan=True):
                    raise ValueError(f"Stale metric summary for {ps.path}: {metric}")
        summary = summary.loc[selected_rows].reset_index(drop=True)
        comparisons = _load_comparisons(
            TABLES_DIR, prediction_sets, summary,
            run_manifest=args.run_manifest or RESULTS_DIR / "run_manifest.json",
        )
    enforce_storage_policy({
        "paper_tables_dir": TABLES_DIR, "paper_figures_dir": FIGURES_DIR,
        "paper_reports_dir": REPORTS_DIR, "paper_bundle_dir": TABLES_DIR.parent,
    })
    metrics_df = prepare_metrics_for_display(summary)
    manifest_path = args.run_manifest or RESULTS_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    metrics_df.attrs["report_notes"] = _report_context_notes(manifest)
    from reporting.manuscript_assets import export_core_comparisons
    export_core_comparisons(metrics_df, TABLES_DIR)

    logger.info("Generating styled tables and reports from %s", metrics_path)
    generate_html_tables(metrics_df)
    generate_markdown_report(metrics_df)
    generate_docx_report(metrics_df)
    generate_pdf_report(metrics_df)
    if comparisons is not None:
        _export_comparison_tables(comparisons, TABLES_DIR, report_notes=metrics_df.attrs["report_notes"])
    statuses = sorted(summary.analysis_status.unique().tolist()) if "analysis_status" in summary else ["legacy"]
    status_record = {"analysis_status": statuses, "configurations": len(summary),
                     "source": "saved validation predictions; no model fitting",
                     "calibration_assessment": "Brier scores and binned held-out reliability curves; held-out calibration intercept/slope and their intervals are not estimated. Saved training calibrator parameters are not evaluation calibration statistics.",
                     "uncertainty_scope": "Patient-cluster resampling conditions on saved fitted-model predictions; it does not repeat model selection or estimate full training-pipeline variability.",
                     "run_manifest": str(args.run_manifest or RESULTS_DIR / "run_manifest.json")}
    if manifest:
        status_record["analysis_role"] = _analysis_role(manifest)
        status_record["branch_cohorts"] = {
            branch: {key: roster[key] for key in ("operations", "patients", "outcomes") if key in roster}
            for branch, roster in manifest.get("branch_rosters", {}).items()
        }
    if comparisons is not None:
        inventory = comparisons["inventory"]
        coverage = pd.Series([row["availability"] for row in inventory["comparisons"]], dtype=object).value_counts()
        delong = comparisons["delong"]
        status_record.update(
            comparison_status=inventory["comparison_status"],
            uncertainty_status=inventory.get("uncertainty_status", "unknown"),
            reporting_policy={key: inventory.get(key) for key in (
                "bootstrap_replicates", "bootstrap_seed", "bootstrap_stratified", "confidence_level")},
            comparison_coverage={str(key): int(value) for key, value in coverage.items()},
            ci_status="nominal pointwise 95% intervals conditional on saved fitted-model predictions",
            delong_status_counts={str(key): int(value) for key, value in delong.get("status", pd.Series(dtype=object)).value_counts().items()},
            dca_status="exploratory; uncertainty is pointwise and does not establish clinical utility",
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "report_status.json").write_text(json.dumps(status_record, indent=2) + "\n")
    report_path = REPORTS_DIR / "report.md"
    if report_path.exists():
        coverage_note = (f"**Comparison coverage: {comparisons['inventory']['comparison_status']}**\n\n"
                         if comparisons is not None else "")
        report_path.write_text("**Analysis status: " + ", ".join(statuses) + "**\n\n"
            + coverage_note + report_path.read_text())

    logger.info("Loading prediction files for plots")
    prediction_frame = _prediction_sets_to_frame(prediction_sets)
    if prediction_frame.empty:
        logger.warning("No prediction files found; skipping plots.")
    else:
        plot_config = PlotConfig.from_env()
        if plot_config.feature_set_allowlist is None:
            from reporting.feature_set_display import CORE_FEATURE_SET_ORDER
            plot_config.feature_set_allowlist = set(CORE_FEATURE_SET_ORDER)
        plot_curves(prediction_frame, config=plot_config)
        if comparisons is not None:
            from reporting.decision_curve import render_comparison_curves
            render_comparison_curves(
                comparisons["dca"], comparisons["dca_comparisons"], FIGURES_DIR,
                comparison_status=comparisons["inventory"]["comparison_status"],
            )
        else:
            from reporting.decision_curve import export_decision_curves
            export_decision_curves(prediction_frame.loc[prediction_frame.feature_set.isin(plot_config.feature_set_allowlist)], FIGURES_DIR)
        if args.explain_folds:
            from reporting.fold_explanations import export_fold_explanations
            export_fold_explanations(RESULTS_DIR, FIGURES_DIR / "fold_explanations")
        from reporting.manuscript_assets import refresh_paper_bundle
        refresh_paper_bundle(TABLES_DIR.parent)
        logger.info("Plots saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
