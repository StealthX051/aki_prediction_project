"""Exploratory decision curves from locked saved out-of-fold probabilities."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reporting.feature_set_display import CORE_FEATURE_SET_ORDER, feature_set_sort_key
from reporting.display_dictionary import load_display_dictionary
from reporting.figure_rendering import feature_set_color, report_figure_style_context, save_report_figure_bundle

DEFAULT_RANGES = {"any_aki": (0.01, 0.30), "icu_admission": (0.05, 0.60), "mortality": (0.001, 0.050)}
# Integer grids avoid rounding rare-outcome thresholds to zero.
_GRID_INTEGERS = {"any_aki": (1, 30, 100), "icu_admission": (5, 60, 100), "mortality": (1, 50, 1000)}
FAMILY_ORDER = ("logistic_regression", "xgboost", "ebm")
FAMILY_COLORS = {"logistic_regression": "#2d6ba3", "xgboost": "#bf3b3b", "ebm": "#2f855a"}
FAMILY_LABELS = {"logistic_regression": "LR", "xgboost": "XGBoost", "ebm": "EBM"}


def outcome_thresholds(outcome: str) -> np.ndarray:
    """The prespecified exploratory grid, including both decimal boundaries."""
    if outcome not in DEFAULT_RANGES:
        raise ValueError(f"No exploratory DCA range for {outcome!r}")
    lower, upper, denominator = _GRID_INTEGERS[outcome]
    return np.arange(lower, upper + 1, dtype=float) / denominator


def threshold_grid_metadata(thresholds) -> dict:
    cutoffs = np.asarray(thresholds, dtype=float)
    steps = np.diff(cutoffs)
    step = float(steps[0]) if len(steps) and np.allclose(steps, steps[0]) else None
    return {"threshold_min": float(cutoffs.min()), "threshold_max": float(cutoffs.max()),
            "threshold_step": step, "threshold_probabilities": cutoffs.tolist(),
            "threshold_grid_hash": hashlib.sha256(json.dumps(cutoffs.tolist(), separators=(",", ":")).encode()).hexdigest()}


def _patient_codes(groups):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups) or pd.isna(groups).any():
        raise ValueError("Require a nonempty vector of nonmissing patient identities")
    codes, patients = pd.factorize(groups, sort=False)
    return codes, len(patients)


def cluster_resample_counts(groups, indices_list) -> np.ndarray:
    """Convert whole-patient row draws to first-appearance patient multiplicities."""
    codes, n_patients = _patient_codes(groups)
    first = np.full(n_patients, len(codes), dtype=int)
    np.minimum.at(first, codes, np.arange(len(codes)))
    counts = np.empty((len(indices_list), n_patients), dtype=np.int32)
    for row, indices in enumerate(indices_list):
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
                or (indices < 0).any() or (indices >= len(codes)).any()):
            raise ValueError("Bootstrap indices must be nonempty integer row indices in range")
        multiplicities = np.bincount(indices, minlength=len(codes))
        patient_counts = multiplicities[first]
        if not np.array_equal(multiplicities, patient_counts[codes]):
            raise ValueError("Bootstrap draws must retain all operations of each sampled patient")
        if int(patient_counts.sum()) != n_patients:
            raise ValueError("Each bootstrap draw must sample the original number of patients")
        counts[row] = patient_counts
    return counts


def net_benefit(y_true, y_prob, thresholds) -> pd.DataFrame:
    """Operation-level net benefit; thresholds are exploratory, never model tuning."""
    truth = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=float)
    cutoffs = np.asarray(thresholds, dtype=float)
    if truth.ndim != 1 or len(truth) == 0 or truth.shape != probabilities.shape:
        raise ValueError("Require nonempty aligned outcome and probability vectors")
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Require binary outcomes and finite probabilities in [0,1]")
    if cutoffs.ndim != 1 or len(cutoffs) == 0 or not np.isfinite(cutoffs).all() or ((cutoffs <= 0) | (cutoffs >= 1)).any():
        raise ValueError("Threshold probabilities must lie strictly between zero and one")
    positive = probabilities[:, None] >= cutoffs[None, :]
    tp = ((truth[:, None] == 1) & positive).sum(axis=0)
    fp = ((truth[:, None] == 0) & positive).sum(axis=0)
    odds = cutoffs / (1 - cutoffs)
    prevalence = truth.mean()
    return pd.DataFrame({"threshold_probability": cutoffs,
                         "model": tp / len(truth) - fp / len(truth) * odds,
                         "treat_all": prevalence - (1 - prevalence) * odds,
                         "treat_none": np.zeros(len(cutoffs))})


def bootstrap_net_benefit(y_true, y_prob, thresholds, *, groups, cluster_counts):
    """Pointwise patient-bootstrap uncertainty, conditional on locked predictions.

    Patient totals are aggregated once; batched matrix products then retain the
    operation-level estimand even when patients contribute different row counts.
    Returned draws have shape ``(replicates, thresholds)`` and can be subtracted
    directly when both models use the same ``cluster_counts``.
    """
    table = net_benefit(y_true, y_prob, thresholds)
    truth, probabilities = np.asarray(y_true), np.asarray(y_prob, dtype=float)
    codes, n_patients = _patient_codes(groups)
    if len(codes) != len(truth):
        raise ValueError("Patient identities must align with outcome/probability vectors")
    counts = np.asarray(cluster_counts)
    if (counts.ndim != 2 or counts.shape[1] != n_patients or counts.dtype.kind not in "iu"
            or (counts < 0).any() or (counts.sum(axis=1) != n_patients).any()):
        raise ValueError("Require nonnegative integer whole-patient bootstrap counts")
    cutoffs = table.threshold_probability.to_numpy()
    odds = cutoffs / (1 - cutoffs)
    selected = probabilities[:, None] >= cutoffs[None, :]
    contributions = selected * (truth[:, None] - (1 - truth[:, None]) * odds)
    patient_contributions = np.zeros((n_patients, len(cutoffs)))
    np.add.at(patient_contributions, codes, contributions)
    patient_positives = np.bincount(codes, weights=truth, minlength=n_patients)
    patient_sizes = np.bincount(codes, minlength=n_patients)
    model_draws = np.empty((len(counts), len(cutoffs)))
    all_draws = np.empty_like(model_draws)
    for start in range(0, len(counts), 128):
        batch = counts[start:start + 128]
        n_operations = batch @ patient_sizes
        model_draws[start:start + len(batch)] = (batch @ patient_contributions) / n_operations[:, None]
        prevalence = (batch @ patient_positives) / n_operations
        all_draws[start:start + len(batch)] = prevalence[:, None] - (1 - prevalence[:, None]) * odds
    for name, draws in (("model", model_draws), ("treat_all", all_draws)):
        if len(counts) >= 2:
            lower, upper = np.percentile(draws, [2.5, 97.5], axis=0)
        else:
            lower = upper = np.full(len(cutoffs), np.nan)
        table[f"{name}_lower"], table[f"{name}_upper"] = lower, upper
        table[f"{name}_bootstrap_valid"] = len(counts)
    return table, model_draws


def export_decision_curves(predictions: pd.DataFrame, output_dir: Path) -> list[Path]:
    for identity in ("cohort_scope", "validation_scheme", "analysis_status"):
        if identity in predictions and predictions[identity].nunique(dropna=False) != 1:
            raise ValueError(f"Decision curves cannot mix {identity}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    labels = load_display_dictionary().feature_set_labels()
    for (outcome, branch, family), group in predictions.groupby(["outcome", "branch", "model_name"]):
        if outcome not in DEFAULT_RANGES:
            continue
        lower, upper = DEFAULT_RANGES[outcome]
        thresholds = outcome_thresholds(outcome)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        tables = []
        reference_identity = None
        for feature_set in sorted(group["feature_set"].unique(), key=feature_set_sort_key):
            rows = group.loc[group["feature_set"].eq(feature_set)].sort_values("caseid")
            if rows["caseid"].duplicated().any():
                raise ValueError("Decision curves require unique operation IDs")
            identity_columns = [c for c in ("caseid", "subjectid", "y_true", "outer_fold_id", "repeat_id") if c in rows]
            identity = rows[identity_columns].reset_index(drop=True)
            if reference_identity is not None and not identity.equals(reference_identity):
                raise ValueError("Decision-curve comparisons require matched operation/patient/outcome/fold rosters")
            reference_identity = identity
            table = net_benefit(rows["y_true"], rows["y_prob_calibrated"], thresholds)
            table["feature_set"] = feature_set
            tables.append(table)
            ax.plot(thresholds, table["model"], label=labels.get(feature_set, feature_set), color=feature_set_color(feature_set))
        reference = tables[0]
        ax.plot(thresholds, reference["treat_all"], "--", color="grey", label="Treat all")
        ax.axhline(0, color="black", linewidth=1, label="Treat none")
        status = f" [{group['analysis_status'].iloc[0]}]" if "analysis_status" in group else ""
        ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(lower, upper),
               title=f"Exploratory decision curves — {outcome}, {branch}, {family}{status}")
        ax.legend()
        stem = output_dir / f"dca_{outcome}_{branch}_{family}"
        paths.extend(save_report_figure_bundle(fig, stem, formats=("png", "svg"), close=True))
        table_path = stem.with_suffix(".csv")
        pd.concat(tables, ignore_index=True).to_csv(table_path, index=False)
        paths.append(table_path)
        metadata = {"outcome": outcome, "branch": branch, "model_name": family,
                    **threshold_grid_metadata(thresholds),
                    "source": "saved calibrated validation predictions", "n_operations": len(group) // group['feature_set'].nunique(),
                    "interpretation": "Exploratory net benefit does not establish clinical utility or define an intervention."}
        for column in ("cohort_scope", "validation_scheme", "analysis_status"):
            if column in group:
                metadata[column] = group[column].iloc[0]
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return paths


def _saved_curve_metadata(frame, *, outcome, branch, comparison_status, plotted_feature_sets=None):
    metadata = {
        "outcome": outcome, "branch": branch, "comparison_status": comparison_status,
        "source": "saved calibrated validation predictions",
        "uncertainty": "95% pointwise patient-cluster percentile intervals, conditional on fitted predictions",
        "interpretation": "Exploratory net benefit does not establish clinical utility or define an intervention.",
    }
    for column in ("cohort_scope", "validation_scheme", "analysis_status", "n_operations", "n_patients",
                   "bootstrap_replicates", "bootstrap_seed", "bootstrap_stratified", "uncertainty_protocol_version"):
        if column in frame and frame[column].nunique(dropna=False) == 1:
            value = frame[column].iloc[0]
            metadata[column] = value.item() if isinstance(value, np.generic) else value
    identity = [column for column in frame if column.endswith("_hash")]
    if identity:
        identity = [column for column in ("model_type", "feature_set", "comparison_id") if column in frame] + identity
        metadata["prediction_provenance"] = frame[identity].drop_duplicates().to_dict("records")
    if plotted_feature_sets is None:
        feature_columns = [column for column in ("feature_set", "reference_feature_set", "candidate_feature_set") if column in frame]
        plotted_feature_sets = set().union(*(set(frame[column]) for column in feature_columns))
    metadata["plotted_feature_sets"] = sorted(plotted_feature_sets, key=feature_set_sort_key)
    thresholds = np.sort(frame.threshold_probability.unique())
    metadata.update(threshold_grid_metadata(thresholds))
    return metadata


def _save_saved_curve(fig, stem, frame, *, outcome, branch, comparison_status, plotted_feature_sets=None):
    paths = save_report_figure_bundle(fig, stem, formats=("png", "svg"), close=True)
    table_path, metadata_path = stem.with_suffix(".csv"), stem.with_suffix(".json")
    frame.to_csv(table_path, index=False)
    metadata_path.write_text(json.dumps(_saved_curve_metadata(
        frame, outcome=outcome, branch=branch, comparison_status=comparison_status,
        plotted_feature_sets=plotted_feature_sets), indent=2) + "\n")
    return [*paths, table_path, metadata_path]


def render_comparison_curves(
    dca_summary: pd.DataFrame, dca_comparisons: pd.DataFrame, output_dir: Path,
    *, comparison_status: str = "complete",
) -> list[Path]:
    """Render saved absolute and paired decision curves without statistical refits."""
    if dca_summary.empty:
        if not dca_comparisons.empty:
            raise ValueError("DCA comparisons require their saved absolute curves")
        return []
    required = {"outcome", "branch", "model_name", "model_type", "feature_set", "threshold_probability",
                "model", "treat_all", "treat_none"}
    if not required.issubset(dca_summary):
        raise ValueError(f"Missing saved DCA columns: {sorted(required - set(dca_summary))}")
    if not set(dca_summary.outcome).issubset(DEFAULT_RANGES):
        raise ValueError("Saved DCA outcomes require declared exploratory ranges")
    for column in ("cohort_scope", "validation_scheme", "analysis_status"):
        if column in dca_summary and dca_summary[column].nunique(dropna=False) != 1:
            raise ValueError(f"Decision curves cannot mix {column}")
        if (not dca_comparisons.empty and column in dca_summary and column in dca_comparisons
                and set(dca_comparisons[column]) != set(dca_summary[column])):
            raise ValueError(f"Saved DCA comparisons disagree on {column}")
    if dca_summary.duplicated(["outcome", "branch", "model_type", "feature_set", "threshold_probability"]).any():
        raise ValueError("Duplicate saved DCA thresholds within a model/input configuration")
    thresholds = dca_summary.threshold_probability.to_numpy()
    if (not np.isfinite(thresholds).all() or ((thresholds <= 0) | (thresholds >= 1)).any()
            or not np.isfinite(dca_summary[["model", "treat_all", "treat_none"]].to_numpy()).all()):
        raise ValueError("Saved DCA point estimates and thresholds must be finite and valid")
    if not dca_comparisons.empty:
        paired_columns = {"comparison_id", "outcome", "branch", "reference_model", "candidate_model",
                          "reference_feature_set", "candidate_feature_set", "threshold_probability", "delta", "lower", "upper"}
        if not paired_columns.issubset(dca_comparisons):
            raise ValueError("Missing saved paired DCA columns")
        if dca_comparisons.duplicated(["comparison_id", "threshold_probability"]).any():
            raise ValueError("Duplicate saved paired DCA thresholds")
        if not np.isfinite(dca_comparisons[["threshold_probability", "delta"]].to_numpy()).all():
            raise ValueError("Saved paired DCA point estimates must be finite")
    output_dir.mkdir(parents=True, exist_ok=True)
    display = load_display_dictionary()
    labels, paths = display.feature_set_labels(), []
    status = f" [{dca_summary.analysis_status.iloc[0]}]" if "analysis_status" in dca_summary else ""
    if comparison_status != "complete":
        status += " [partial comparisons]"
    with report_figure_style_context():
        absolute_groups = list(dca_summary.groupby(["outcome", "branch", "model_name"], sort=False))
        absolute_groups.sort(key=lambda pair: (*pair[0][:2], _family_sort_key(pair[1].model_type.iloc[0])))
        for (outcome, branch, family), group in absolute_groups:
            plotted_features = [feature for feature in CORE_FEATURE_SET_ORDER if feature in set(group.feature_set)]
            if not plotted_features:
                continue
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            for feature in plotted_features:
                rows = group.loc[group.feature_set.eq(feature)].sort_values("threshold_probability")
                ax.plot(rows.threshold_probability, rows.model, color=feature_set_color(feature), label=labels.get(feature, feature))
            _plot_default_strategies(ax, rows)
            ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=DEFAULT_RANGES[outcome])
            ax.set_title(f"{display.outcome_label(outcome)} · {family}\nExploratory decision curves{status}",
                         fontsize=12, fontweight="normal")
            ax.legend()
            paths.extend(_save_saved_curve(fig, output_dir / f"dca_{outcome}_{branch}_{family}", group,
                                          outcome=outcome, branch=branch, comparison_status=comparison_status,
                                          plotted_feature_sets=plotted_features))
        family_groups = list(dca_summary.loc[dca_summary.feature_set.isin(CORE_FEATURE_SET_ORDER)].groupby(
            ["outcome", "branch", "feature_set"], sort=False))
        family_groups.sort(key=lambda pair: (*pair[0][:2], feature_set_sort_key(pair[0][2])))
        for (outcome, branch, feature), group in family_groups:
            if group.model_type.nunique() < 2:
                continue
            fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
            for family in FAMILY_ORDER:
                rows = group.loc[group.model_type.eq(family)].sort_values("threshold_probability")
                if not rows.empty:
                    ax.plot(rows.threshold_probability, rows.model, color=FAMILY_COLORS[family], label=FAMILY_LABELS[family])
            _plot_default_strategies(ax, group.loc[group.model_type.eq(group.model_type.iloc[0])].sort_values("threshold_probability"))
            ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=DEFAULT_RANGES[outcome])
            ax.set_title(f"{display.outcome_label(outcome)} · {labels.get(feature, feature)}\nExploratory model comparison{status}",
                         fontsize=12, fontweight="normal")
            ax.legend()
            paths.extend(_save_saved_curve(fig, output_dir / f"dca_families_{outcome}_{branch}_{feature}", group,
                                          outcome=outcome, branch=branch, comparison_status=comparison_status))
        if not dca_comparisons.empty:
            core = dca_comparisons.loc[dca_comparisons.reference_feature_set.isin(CORE_FEATURE_SET_ORDER)
                                      & dca_comparisons.candidate_feature_set.isin(CORE_FEATURE_SET_ORDER)].copy()
            core["_panel"] = np.where(core.reference_model.eq(core.candidate_model), core.candidate_model,
                                      "families_" + core.candidate_feature_set)
            delta_groups = list(core.groupby(["outcome", "branch", "_panel"], sort=False))
            delta_groups.sort(key=lambda pair: (*pair[0][:2],
                (1, feature_set_sort_key(pair[0][2][len("families_"):])) if pair[0][2].startswith("families_")
                else (0, _family_sort_key(pair[0][2]))))
            for (outcome, branch, panel), group in delta_groups:
                comparisons = sorted(group.drop_duplicates("comparison_id").itertuples(), key=lambda row: (
                    _family_sort_key(row.reference_model), _family_sort_key(row.candidate_model),
                    feature_set_sort_key(row.reference_feature_set), feature_set_sort_key(row.candidate_feature_set)))
                fig, axes = plt.subplots(len(comparisons), 1, figsize=(8, 3.1 * len(comparisons)),
                                         squeeze=False, constrained_layout=True, sharex=True)
                for ax, identity in zip(axes[:, 0], comparisons):
                    rows = group.loc[group.comparison_id.eq(identity.comparison_id)].sort_values("threshold_probability")
                    color = FAMILY_COLORS.get(identity.candidate_model, "#2f855a")
                    ax.plot(rows.threshold_probability, rows.delta, color=color)
                    finite = np.isfinite(rows.lower) & np.isfinite(rows.upper)
                    ax.fill_between(rows.threshold_probability.to_numpy(), rows.lower.to_numpy(), rows.upper.to_numpy(),
                                    where=finite.to_numpy(), color=color, alpha=0.16)
                    ax.axhline(0, color="#52606d", linewidth=0.8)
                    if identity.candidate_model == identity.reference_model:
                        left = labels.get(identity.candidate_feature_set, identity.candidate_feature_set)
                        right = labels.get(identity.reference_feature_set, identity.reference_feature_set)
                    else:
                        left = FAMILY_LABELS[identity.candidate_model]
                        right = FAMILY_LABELS[identity.reference_model]
                    ax.set(ylabel="Δ net benefit", xlim=DEFAULT_RANGES[outcome])
                    ax.set_title(f"{left} − {right}", fontsize=11, fontweight="normal")
                axes[-1, 0].set_xlabel("Threshold probability")
                panel_label = labels.get(panel[len("families_"):], panel) if panel.startswith("families_") else FAMILY_LABELS.get(panel, panel)
                fig.suptitle(f"{display.outcome_label(outcome)} · {panel_label}: net-benefit differences\n"
                             f"Exploratory · 95% pointwise intervals{status}", fontsize=11, fontweight="normal")
                paths.extend(_save_saved_curve(fig, output_dir / f"dca_delta_{outcome}_{branch}_{panel}",
                                              group.drop(columns="_panel"), outcome=outcome, branch=branch,
                                              comparison_status=comparison_status))
    return paths


def _plot_default_strategies(ax, rows):
    ax.plot(rows.threshold_probability, rows.treat_all, "--", color="grey", label="Treat all")
    ax.axhline(0, color="black", linewidth=1, label="Treat none")


def _family_sort_key(family):
    return (FAMILY_ORDER.index(family), family) if family in FAMILY_ORDER else (len(FAMILY_ORDER), family)
