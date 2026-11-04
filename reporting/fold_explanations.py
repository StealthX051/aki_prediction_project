"""Regenerate explanations from saved outer-fold bundles without model fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_creation import utils
from artifact_paths import get_scoped_results_dir, get_scoped_paper_dir
from reporting.figure_rendering import save_report_figure_bundle, report_figure_style_context
from reporting.publication_style import color
from data_preparation.artifact_metadata import file_sha256
from reporting.display_dictionary import load_display_dictionary


def _feature_labels(features):
    dictionary = load_display_dictionary()
    return {str(name): dictionary.feature_label(str(name), use_short=True, include_unit=True) for name in features}


def _bar(table: pd.DataFrame, value: str, destination: Path, title: str):
    selected = table.loc[table[value].abs().nlargest(20).index].iloc[::-1]
    display = _feature_labels(selected["feature"])
    labels = [textwrap.fill(display[str(feature)], width=45) for feature in selected["feature"]]
    with report_figure_style_context():
        fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * sum(label.count("\n") + 1 for label in labels))), constrained_layout=True)
        ax.barh(labels, selected[value], color=color('ink'))
        xlabel = "Raw log-odds coefficient (numeric inputs per training SD)" if value == "coefficient" else value
        ax.set(title=title, xlabel=xlabel)
        return save_report_figure_bundle(fig, destination, close=True)


def export_fold_bundle(fold_dir: Path, raw: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Explain only the held-out operation roster belonging to one fitted fold."""
    metadata = json.loads((fold_dir / "metadata.json").read_text())
    if metadata.get("status", "completed") != "completed":
        raise ValueError("Cannot explain an incomplete fold checkpoint")
    expected_hash = metadata.get("validation_fingerprint", {}).get("dataset_hash")
    actual_hash = raw.attrs.get("source_dataset_sha256")
    if not expected_hash or not actual_hash or actual_hash != expected_hash:
        raise ValueError("Explanation dataset hash differs from the fitted modeling CSV or its provenance is missing")
    if metadata.get("checkpoint_bundle_protocol_version") == 2:
        from model_creation.validation import is_outer_fold_checkpoint_complete
        if not is_outer_fold_checkpoint_complete(fold_dir):
            raise ValueError("Explanation requires a complete verified outer-fold bundle")
        metadata["bundle_integrity"] = "all saved fold artifacts verified"
    else:
        metadata["bundle_integrity"] = "legacy bundle: model/preprocessor hashes checked when present; full prediction/calibration integrity unavailable"
    family = metadata["model_type"]
    title_prefix = "Diagnostic: " if metadata.get("validation_fingerprint", {}).get("diagnostic") is True else ""
    model_name = "model.json" if family == "xgboost" else "model.pkl"
    for filename, key in ((model_name, "model_sha256"), ("preprocessor.pkl", "preprocessor_sha256")):
        if key in metadata and hashlib.sha256((fold_dir / filename).read_bytes()).hexdigest() != metadata[key]:
            raise ValueError(f"Saved explanation bundle hash mismatch: {filename}")
    predictions = pd.read_csv(fold_dir / "test_predictions.csv")
    if raw["caseid"].duplicated().any() or predictions["caseid"].duplicated().any():
        raise ValueError("Explanation inputs require unique operation IDs")
    lookup = raw.set_index("caseid")
    if not predictions["caseid"].isin(lookup.index).all():
        raise ValueError("Held-out explanation case missing from branch dataset")
    with (fold_dir / "preprocessor.pkl").open("rb") as handle:
        preprocessor = pickle.load(handle)
    columns = list(preprocessor.numeric_columns_) + list(preprocessor.categorical_columns_)
    aligned = lookup.loc[predictions["caseid"], columns]
    transformed = preprocessor.transform(aligned)
    display_labels = _feature_labels(transformed.columns)
    if family == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(fold_dir / "model.json")
    else:
        with (fold_dir / "model.pkl").open("rb") as handle:
            model = pickle.load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if family == "logistic_regression":
        names = list(transformed.columns)
        coefficients = pd.DataFrame({"feature": names, "coefficient": model.coef_[0],
            "display_label": [display_labels[name] for name in names],
            "interpretation": ["per training SD" if name in preprocessor.numeric_columns_
                               else "encoded category or missingness indicator" for name in names]})
        path = output_dir / "coefficients.csv"
        coefficients.to_csv(path, index=False)
        paths.append(path)
        paths.extend(_bar(coefficients, "coefficient", output_dir / "coefficients", title_prefix + "Outer-fold logistic regression coefficients"))
        metadata["intercept"] = float(model.intercept_[0])
    elif family == "xgboost":
        import shap
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(transformed))
        metadata["shap_base_value"] = np.asarray(explainer.expected_value).tolist()
        metadata["shap_output_scale"] = "raw model log-odds; base value plus feature contributions reconstructs the uncalibrated margin"
        contributions = pd.DataFrame(values, columns=transformed.columns)
        contributions.insert(0, "caseid", predictions["caseid"].to_numpy())
        path = output_dir / "heldout_shap_values.csv"
        contributions.to_csv(path, index=False)
        paths.append(path)
        importance = pd.DataFrame({"feature": transformed.columns, "mean_absolute_SHAP": np.abs(values).mean(axis=0)})
        importance["display_label"] = importance.feature.map(display_labels)
        importance.to_csv(output_dir / "shap_importance.csv", index=False)
        paths.extend(_bar(importance, "mean_absolute_SHAP", output_dir / "shap_importance", title_prefix + "Held-out outer-fold SHAP"))
        with report_figure_style_context():
            shap.summary_plot(values, transformed.rename(columns=display_labels), show=False, max_display=20)
            if title_prefix:
                plt.gcf().suptitle(title_prefix + "Held-out outer-fold SHAP")
            paths.extend(save_report_figure_bundle(plt.gcf(), output_dir / "shap_beeswarm", close=True))
    elif family == "ebm":
        explanation = model.explain_global()
        global_data = explanation.data()
        importance = pd.DataFrame({"feature": model.term_names_, "importance": model.term_importances()})
        importance["display_label"] = importance.feature.map(_feature_labels(model.term_names_))
        importance.to_csv(output_dir / "term_importance.csv", index=False)
        paths.extend(_bar(importance, "importance", output_dir / "term_importance", title_prefix + "Outer-fold EBM native term importance"))
        local = pd.DataFrame(model.eval_terms(transformed), columns=model.term_names_)
        local.insert(0, "caseid", predictions["caseid"].to_numpy())
        local.to_csv(output_dir / "heldout_term_contributions.csv", index=False)
        payload = {"global": global_data, "terms": [],
                   "interpretation": "Native response curves show observed-value bins. Missing and unknown bins are exported separately in special_bins and special_bin_scores.csv; scores are raw model log-odds contributions, not causal effects."}
        special_bins = []
        for idx, term in enumerate(model.term_names_):
            term_data = explanation.data(idx)
            term_special = []
            raw_scores = np.asarray(model.term_scores_[idx])
            if raw_scores.ndim == 1:
                weights = np.asarray(model.bin_weights_[idx])
                for bin_type, position in (("missing", 0), ("unknown", -1)):
                    row = {"term": str(term), "bin_type": bin_type,
                           "raw_log_odds_contribution": float(raw_scores[position]),
                           "training_weight": float(weights[position])}
                    term_special.append(row)
                    special_bins.append(row)
            payload["terms"].append({"term": term, "data": term_data, "special_bins": term_special})
            scores = np.asarray(term_data.get("scores", []))
            names = np.asarray(term_data.get("names", []))
            if scores.ndim != 1 or len(scores) == 0:
                continue
            x = np.arange(len(scores))
            with report_figure_style_context():
                fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
                if len(names) == len(scores) + 1 and np.issubdtype(names.dtype, np.number):
                    x = (names[:-1].astype(float) + names[1:].astype(float)) / 2
                ax.plot(x, scores, color=color('ink'))
                ax.set(title=title_prefix + _feature_labels([term])[str(term)], xlabel="Feature value / encoded category index", ylabel="Raw model log-odds contribution")
                paths.extend(save_report_figure_bundle(fig, output_dir / f"response_{idx:03d}", close=True))
        special_path = output_dir / "special_bin_scores.csv"
        pd.DataFrame(special_bins, columns=["term", "bin_type", "raw_log_odds_contribution", "training_weight"]).to_csv(special_path, index=False)
        paths.append(special_path)
        (output_dir / "native_explanation.json").write_text(json.dumps(payload, default=lambda value: np.asarray(value).tolist(), indent=2) + "\n")
    else:
        raise ValueError(f"Unsupported explanation family: {family}")
    metadata.update({"source_fold": str(fold_dir), "heldout_operations": len(predictions),
                     "feature_display_labels": display_labels,
                     "caseids": predictions["caseid"].tolist(),
                     "interpretation": "Explanations describe the raw fold estimator; logistic recalibration is separate. Penalized coefficients/categories are fold-specific and are not averaged across folds. One-hot category contrasts require coefficient differences within a fold; individual encoded coefficients are not reference-category odds ratios. Model explanations are associations, not causal effects."})
    (output_dir / "explanation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return paths


def export_fold_explanations(results_dir: Path, output_dir: Path) -> list[Path]:
    paths = []
    datasets = {}
    for fold in sorted(results_dir.glob("models/**/artifacts/folds/repeat_*/outer_*")):
        metadata = json.loads((fold / "metadata.json").read_text())
        if not metadata.get("validation_fingerprint", {}).get("diagnostic"):
            from model_creation.validation import is_outer_fold_checkpoint_complete
            if not is_outer_fold_checkpoint_complete(fold):
                raise ValueError("Definitive explanations require a complete verified outer-fold bundle")
        branch = metadata.get("branch") or metadata["validation_fingerprint"]["branch"]
        if branch not in datasets:
            source = {"non_windowed": utils.FULL_FEATURES_FILE,
                      "windowed": utils.WINDOWED_FEATURES_FILE}.get(branch)
            if source is None:
                raise ValueError(f"Unsupported explanation branch: {branch}")
            source_hash = file_sha256(source)
            if source_hash != metadata.get("validation_fingerprint", {}).get("dataset_hash"):
                raise ValueError("Explanation dataset hash differs from the fitted modeling CSV")
            datasets[branch] = utils.load_data(branch)
            # Detect replacement while reading rather than explain a changed file.
            if file_sha256(source) != source_hash:
                raise ValueError("Explanation modeling CSV changed while loading")
            datasets[branch].attrs["source_dataset_sha256"] = source_hash
        relative = fold.relative_to(results_dir / "models")
        paths.extend(export_fold_bundle(fold, datasets[branch], output_dir / relative))
    if not paths:
        raise ValueError("No saved outer-fold explanation bundles found")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=get_scoped_results_dir(Path(__file__).resolve().parents[1], os.getenv("COHORT_SCOPE", "main_shared"), os.getenv("VALIDATION_SCHEME", "nested_cv")))
    parser.add_argument("--output-dir", type=Path, default=get_scoped_paper_dir(Path(__file__).resolve().parents[1], os.getenv("COHORT_SCOPE", "main_shared"), os.getenv("VALIDATION_SCHEME", "nested_cv")) / "figures" / "fold_explanations")
    args = parser.parse_args()
    export_fold_explanations(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
