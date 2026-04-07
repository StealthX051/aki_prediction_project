import argparse
import concurrent.futures
import hashlib
import json
import logging
import time
import os
import re
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_creation import utils
from model_creation.postprocessing import (
    LogisticRecalibrationModel,
    write_json,
)
from model_creation.prediction_io import write_prediction_files
from model_creation.validation import (
    VALIDATION_PROTOCOL_VERSION,
    build_cv_splits,
    build_validation_fingerprint,
    checkpoint_matches_validation_fingerprint,
    fit_final_model,
    generate_cross_fitted_predictions,
    get_effective_refit_params,
    get_refit_param_overrides,
    is_outer_fold_checkpoint_complete,
    load_outer_fold_checkpoint,
    run_outer_split,
    save_model_bundle,
    save_outer_fold_checkpoint,
    select_modeling_dataset,
    tune_model,
    write_validation_manifest,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PIPELINE_NAME = "step_07_train_evaluate"


def compute_file_hash(path: Path) -> str:
    """Compute the SHA256 hash of a file without loading it into memory."""

    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_shap_plots(model, X_test, output_dir):
    """
    Calculates and saves SHAP summary plots.
    """
    logger.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Bar plot
    plt.figure(figsize=(10, max(10, len(X_test.columns) * 0.5)))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (Bar)")
    plt.savefig(output_dir / "shap_summary_bar.png", bbox_inches='tight')
    plt.close()

    # Dot plot
    plt.figure(figsize=(10, max(10, len(X_test.columns) * 0.5)))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False, max_display=20)
    plt.title("SHAP Summary Plot (Dot)")
    plt.savefig(output_dir / "shap_summary_dot.png", bbox_inches='tight')
    plt.close()
    
    logger.info("SHAP plots saved.")


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _json_safe(data):
    return json.loads(json.dumps(data, default=_json_default))


def _safe_logit(probabilities: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    clipped = np.clip(probabilities, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def _safe_filename(name: str) -> str:
    """Coerce an arbitrary feature name into a filesystem-friendly slug."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return slug or "feature"


def _assert_nested_test_prediction_coverage(
    test_predictions: pd.DataFrame,
    expected_caseids: pd.Series,
) -> None:
    duplicated = test_predictions["caseid"].duplicated()
    if duplicated.any():
        dupes = sorted(test_predictions.loc[duplicated, "caseid"].unique().tolist())
        raise ValueError(
            "Nested pooled test predictions contain duplicate caseids; expected exactly one "
            f"evaluation row per operation. Duplicate caseids: {dupes[:10]}"
        )

    actual_caseids = set(test_predictions["caseid"].tolist())
    expected_caseid_set = set(expected_caseids.tolist())
    if actual_caseids != expected_caseid_set:
        missing = sorted(expected_caseid_set - actual_caseids)
        extra = sorted(actual_caseids - expected_caseid_set)
        raise ValueError(
            "Nested pooled test predictions do not cover the modeling cohort exactly once. "
            f"Missing caseids: {missing[:10]}; extra caseids: {extra[:10]}"
        )

    if len(test_predictions) != len(expected_caseids):
        raise ValueError(
            "Nested pooled test predictions must have exactly one row per modeled operation. "
            f"Expected {len(expected_caseids)} rows, found {len(test_predictions)}."
        )


def _ebm_term_metadata(model) -> Tuple[List[str], List[float]]:
    """Return (term_names, term_importances) with best-effort fallbacks across interpret versions."""
    names: List[str] = []
    raw_names = getattr(model, "term_names_", None)
    if not raw_names:
        raw_names = getattr(model, "term_names", None)
        if callable(raw_names):
            try:
                raw_names = raw_names()
            except Exception:
                raw_names = None
    if raw_names:
        try:
            names = [str(name) for name in raw_names]
        except Exception:
            names = list(raw_names)

    importances: List[float] = []
    raw_importances = getattr(model, "term_importances_", None)
    if raw_importances is None or (hasattr(raw_importances, "__len__") and len(raw_importances) == 0):
        raw_importances = getattr(model, "term_importances", None)
        if callable(raw_importances):
            try:
                raw_importances = raw_importances()
            except Exception:
                raw_importances = None

    if raw_importances is not None:
        try:
            importances = np.asarray(raw_importances, dtype=float).flatten().tolist()
        except Exception:
            try:
                importances = [float(value) for value in raw_importances]
            except Exception:
                importances = []

    if names and importances and len(names) != len(importances):
        logger.warning(
            "Length mismatch between term names (%s) and importances (%s); clipping to match.",
            len(names),
            len(importances),
        )
        limit = min(len(names), len(importances))
        names = names[:limit]
        importances = importances[:limit]

    return names, importances


def _format_feature_labels(feature_names: Sequence[str], display_dict: Optional[object]) -> List[str]:
    """Map raw feature names to human-readable labels using the display dictionary when available."""
    if display_dict is None:
        return list(feature_names)

    labels: List[str] = []
    for name in feature_names:
        try:
            # Short labels keep plots readable; include units when present.
            label = display_dict.feature_label(name, use_short=True, include_unit=True)
        except Exception:
            label = str(name)
        labels.append(label)
    return labels


def _is_number(value: object) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _write_xai_index(
    xai_dir: Path,
    *,
    global_section: Dict[str, str],
    term_entries: List[Dict[str, str]],
) -> None:
    """Emit a lightweight index.html to navigate EBM XAI artifacts."""

    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>EBM Interpretability Index</title>",
        "<style>",
        "body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 2rem; max-width: 960px; color: #0f172a; }",
        "h1 { margin-bottom: 0.25rem; }",
        "h2 { margin-top: 1.5rem; }",
        "ul { line-height: 1.6; padding-left: 1.1rem; }",
        "li { margin-bottom: 0.2rem; }",
        "a { color: #0b3c5d; text-decoration: none; }",
        "a:hover { text-decoration: underline; }",
        "code { background: #f3f4f6; padding: 0.1rem 0.3rem; border-radius: 3px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>EBM Interpretability Artifacts</h1>",
        "<p>Browse global summaries and per-feature partial plots exported by the pipeline.</p>",
        "<h2>Global summaries</h2>",
        "<ul>",
    ]

    for label, rel_path in global_section.items():
        if rel_path:
            lines.append(f'<li><a href="{rel_path}">{label}</a></li>')
    lines.append("</ul>")

    lines.append("<h2>Per-feature partial plots</h2>")
    lines.append("<ul>")
    for entry in term_entries:
        label = entry.get("display_name", entry.get("slug", "feature"))
        rel = entry.get("plot_html")
        raw = entry.get("raw_name")
        extra = f" <code>{raw}</code>" if raw else ""
        if rel:
            lines.append(f'<li><a href="{rel}">{label}</a>{extra}</li>')
        else:
            lines.append(f"<li>{label}{extra}</li>")
    lines.extend(["</ul>", "</body>", "</html>"])

    (xai_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def _export_plotly_xy(
    x_values: Sequence,
    y_values: Sequence[float],
    destination: Path,
    title: str,
    *,
    x_label: str,
    y_label: str,
    categorical: bool = False,
    density: Optional[object] = None,
) -> None:
    """Export a line/bar chart for a single feature."""

    try:
        import plotly.graph_objects as go  # type: ignore
        from plotly.subplots import make_subplots  # type: ignore
    except ImportError:
        logger.warning("Plotly is not installed; skipping Plotly export for %s", title)
        return

    x_list = list(x_values)
    y_list = list(y_values)

    if not x_list or not y_list:
        logger.warning("No data available for Plotly export of %s", title)
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    density_x: Optional[List] = None
    density_y: Optional[List[float]] = None
    if isinstance(density, dict):
        dx = density.get("names")
        dy = density.get("scores")
        if dx is not None and dy is not None:
            try:
                density_x = list(dx)
                density_y = [float(v) for v in dy]
            except Exception:
                density_x = None
                density_y = None
    elif density is not None:
        try:
            candidate = [float(v) for v in density]  # type: ignore
            density_x = x_list
            density_y = candidate
        except Exception:
            density_x = None
            density_y = None

    if density_y:
        fig.add_bar(
            x=density_x or x_list,
            y=density_y,
            name="Density",
            opacity=0.35,
            marker_color="#9aa5b1",
            secondary_y=False,
        )

    if categorical:
        fig.add_bar(
            x=x_list,
            y=y_list,
            name="Contribution",
            marker_color="#1f77b4",
            secondary_y=True,
        )
    else:
        fig.add_scatter(
            x=x_list,
            y=y_list,
            mode="lines+markers",
            name="Contribution",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(color="#1f77b4", size=6),
            secondary_y=True,
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        template="plotly_white",
        font=dict(family="DejaVu Sans", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    if density_y:
        fig.update_yaxes(title_text="Density", secondary_y=False, showgrid=True, gridcolor="#e5e5e5")
    fig.update_yaxes(
        title_text=y_label,
        secondary_y=True,
        zeroline=True,
        zerolinecolor="#cccccc",
        showgrid=True,
        gridcolor="#f0f0f0",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5", tickfont=dict(size=11))

    html_path = destination.with_suffix(".html")
    try:
        fig.write_html(str(html_path))
    except Exception:
        logger.exception("Failed to write Plotly HTML for %s; continuing without it.", title)

    image_path = destination.with_suffix(".png")
    try:
        fig.write_image(str(image_path))
    except ValueError as exc:
        logger.warning(
            "Kaleido is unavailable; saved %s HTML but skipped static image export: %s",
            title,
            exc,
        )
    except Exception:
        logger.exception("Failed to write Plotly image for %s; continuing with HTML only.", title)


def _plot_term_importances(
    term_names: Sequence[str],
    term_importances: Sequence[float],
    destination: Path,
    title: str,
    max_terms: int = 20,
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    sorted_terms = sorted(
        zip(term_names, term_importances), key=lambda item: item[1], reverse=True
    )

    if not sorted_terms:
        logger.warning("No terms available to plot for %s", title)
        return

    display_terms = sorted_terms[:max_terms]
    labels, scores = zip(*display_terms)

    plt.figure(figsize=(10.5, max(6, len(display_terms) * 0.45)))
    bar_color = "#0b3c5d"
    plt.barh(labels[::-1], scores[::-1], color=bar_color)
    plt.xlabel("Importance")
    plt.title(title)
    plt.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(destination, bbox_inches="tight", dpi=200)
    plt.close()


def _plot_local_contributions(local_payload: Dict, destination: Path, max_features: int = 20) -> None:
    names = local_payload.get("display_names") or local_payload.get("names")
    scores = local_payload.get("scores")

    if not names or not scores:
        logger.warning("No local explanation data available to plot.")
        return

    first_scores = scores[0] if isinstance(scores, Iterable) else None
    if not isinstance(first_scores, Iterable):
        logger.warning("Unexpected structure for local scores; skipping plot.")
        return

    paired = list(zip(names, first_scores))
    paired.sort(key=lambda item: abs(item[1]), reverse=True)
    paired = paired[:max_features]

    labels, contributions = zip(*paired)
    max_abs = max(abs(c) for c in contributions) if contributions else 1.0
    cmap = plt.cm.coolwarm
    colors = [cmap(0.5 + 0.5 * (c / max_abs)) for c in contributions]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    plt.figure(figsize=(10.5, max(6, len(labels) * 0.45)))
    plt.barh(labels[::-1], contributions[::-1], color=colors[::-1])
    plt.axvline(0, color="#bbbbbb", linewidth=1)
    plt.xlabel("Contribution")
    plt.title("Top Local Contributions (first sample)")
    plt.grid(axis="x", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(destination, bbox_inches="tight", dpi=200)
    plt.close()


def _export_plotly_bar(
    labels: Sequence[str], values: Sequence[float], destination: Path, title: str
) -> None:
    """Export a horizontal bar chart with Plotly, warning if Kaleido is missing.

    The HTML export is always attempted first so interactive artifacts are
    preserved even when static image dependencies are unavailable.
    """

    label_list = list(labels)
    value_list = list(values)

    if not label_list or not value_list:
        logger.warning("No data available for Plotly export of %s", title)
        return

    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        logger.warning(
            "Plotly is not installed; skipping %s Plotly export while retaining JSON/HTML artifacts.",
            title,
        )
        return

    fig = go.Figure(go.Bar(x=value_list, y=label_list, orientation="h", marker_color="#0b3c5d"))
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        template="plotly_white",
        font=dict(family="DejaVu Sans", size=13),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11))
    fig.update_xaxes(showgrid=True, gridcolor="#e5e5e5", tickfont=dict(size=11))

    html_path = destination.with_suffix(".html")
    try:
        fig.write_html(str(html_path))
    except Exception:
        logger.exception("Failed to write Plotly HTML for %s; continuing without it.", title)

    image_path = destination.with_suffix(".png")
    try:
        fig.write_image(str(image_path))
    except ValueError as exc:
        logger.warning(
            "Kaleido is unavailable; saved %s HTML but skipped static image export: %s",
            title,
            exc,
        )
    except Exception:
        logger.exception("Failed to write Plotly image for %s; continuing with HTML only.", title)


def export_ebm_explanations(
    model,
    artifacts_dir: Path,
    random_state: int,
    local_sample_size: int,
    *,
    X_local: pd.DataFrame,
    y_local: pd.Series,
    caseids: pd.Series,
    raw_probabilities: np.ndarray,
    raw_logits: np.ndarray,
    calibrated_probabilities: np.ndarray,
    threshold: float,
    calibration_model: LogisticRecalibrationModel,
) -> None:
    try:
        # interpret >=0.7.4 moved EBMExplanation into _ebm; older versions used _explain
        try:
            from interpret.glassbox._ebm._ebm import EBMExplanation  # type: ignore
        except ImportError:
            from interpret.glassbox._ebm._explain import EBMExplanation  # type: ignore
    except ImportError as exc:
        logger.error("interpret is required to export EBM explanations: %s", exc)
        return

    display_dict = None
    try:
        from reporting.display_dictionary import load_display_dictionary

        display_dict = load_display_dictionary()
    except Exception as exc:
        logger.warning("Display dictionary unavailable; falling back to raw feature names: %s", exc)

    xai_dir = artifacts_dir / "ebm_xai"
    xai_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating EBM global explanations...")
    global_explanation: EBMExplanation = model.explain_global(name="EBM Global")
    global_payload = _json_safe(global_explanation.data())
    # Attach human-readable labels for convenience in downstream use.
    term_names, term_importances = _ebm_term_metadata(model)
    global_payload["display_names"] = _format_feature_labels(term_names, display_dict)
    write_json(xai_dir / "global_explanation.json", global_payload)

    display_term_names = _format_feature_labels(term_names, display_dict)
    name_to_display = {raw: disp for raw, disp in zip(term_names, display_term_names)}
    _plot_term_importances(
        display_term_names,
        term_importances,
        xai_dir / "global_importances.png",
        title="EBM Term Importances",
    )
    _export_plotly_bar(
        display_term_names,
        term_importances,
        xai_dir / "global_importances_plotly",
        title="EBM Term Importances",
    )

    interactions = []
    term_features: Sequence[Sequence[int]] = getattr(model, "term_features_", []) or []
    if not term_features:
        fallback_term_features = getattr(model, "term_features", None)
        if callable(fallback_term_features):
            try:
                term_features = fallback_term_features()
            except Exception:
                term_features = []

    for name, importance, features in zip(term_names, term_importances, term_features):
        if len(features) <= 1:
            continue
        interactions.append(
            {
                "name": name,
                "display_name": name_to_display.get(name, name),
                "importance": float(importance),
                "features": list(features),
            }
        )

    interactions.sort(key=lambda item: item["importance"], reverse=True)
    write_json(xai_dir / "interaction_importances.json", {"interactions": interactions})
    _plot_term_importances(
        [item["display_name"] for item in interactions],
        [item["importance"] for item in interactions],
        xai_dir / "interaction_importances.png",
        title="EBM Interaction Importances",
    )
    _export_plotly_bar(
        [item["display_name"] for item in interactions],
        [item["importance"] for item in interactions],
        xai_dir / "interaction_importances_plotly",
        title="EBM Interaction Importances",
    )

    if local_sample_size and local_sample_size != len(X_local):
        logger.info(
            "Ignoring local_sample_size=%s; exporting explanations for all %s test rows to match predictions.",
            local_sample_size,
            len(X_local),
        )

    logger.info("Generating EBM local explanations for %s test samples...", len(X_local))
    local_explanation: EBMExplanation = model.explain_local(X_local, y_local)
    local_data = local_explanation.data()
    if local_data is None:
        # For local explanations interpret sets overall=None and stores rows under _internal_obj["specific"]
        logger.info("EBM local explanation returned None for overall; using _internal_obj['specific'] instead.")
        internal_specific = getattr(local_explanation, "_internal_obj", {}).get("specific") or []
        if internal_specific:
            names = internal_specific[0].get("names") or []
            scores = [row.get("scores", []) for row in internal_specific]
            values = [row.get("values", []) for row in internal_specific]
            local_payload = {
                "names": names,
                "scores": scores,
                "values": values,
            }
            # preserve any meta/extra from the first row if available
            for key in ("meta", "extra"):
                if key in internal_specific[0]:
                    local_payload[key] = internal_specific[0][key]
        else:
            local_payload = {}
        local_payload = _json_safe(local_payload)
    else:
        local_payload = _json_safe(local_data)

    # Attach human-readable display names for plotting/tooltips.
    if local_payload.get("names"):
        local_payload["display_names"] = _format_feature_labels(local_payload["names"], display_dict)
    local_payload.update(
        {
            "caseid": _json_safe(caseids.tolist()),
            "raw_logit": _json_safe(raw_logits.tolist()),
            "raw_probability": _json_safe(raw_probabilities.tolist()),
            "calibrated_probability": _json_safe(calibrated_probabilities.tolist()),
            "threshold": float(threshold),
            "predicted_label": _json_safe(
                (calibrated_probabilities >= threshold).astype(int).tolist()
            ),
            "calibration_params": _json_safe(
                {
                    "intercept": calibration_model.intercept,
                    "slope": calibration_model.slope,
                    "eps": calibration_model.eps,
                }
            ),
        }
    )

    names: Sequence[str] = local_payload.get("names")
    if names is None:
        names = []
    scores: Sequence[Sequence[float]] = local_payload.get("scores")
    if scores is None:
        scores = []
    contributions: List[Dict[str, float]] = []
    if names and scores and len(scores) == len(caseids):
        for row in scores:
            contributions.append({name: float(value) for name, value in zip(names, row)})

    reconciled = pd.DataFrame(
        {
            "caseid": list(caseids),
            "raw_logit": list(raw_logits),
            "raw_probability": list(raw_probabilities),
            "calibrated_probability": list(calibrated_probabilities),
            "threshold": threshold,
            "predicted_label": (calibrated_probabilities >= threshold).astype(int),
        }
    )

    if contributions:
        reconciled = pd.concat([reconciled.reset_index(drop=True), pd.DataFrame(contributions)], axis=1)

    write_json(xai_dir / "local_explanations.json", local_payload)
    reconciled.to_csv(xai_dir / "local_attributions.csv", index=False)
    _plot_local_contributions(local_payload, xai_dir / "local_contributions.png")

    # Per-term partial plots (one folder per feature to avoid clutter)
    term_dir = xai_dir / "terms"
    term_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot term data first (interpret objects are not thread-safe).
    term_payloads = []
    for idx, (raw_name, display_name) in enumerate(zip(term_names, display_term_names)):
        term_data = global_explanation.data(idx)
        if not term_data:
            continue
        term_payloads.append((idx, raw_name, display_name, _json_safe(term_data)))

    index_terms: List[Dict[str, str]] = []

    def _export_term(payload):
        idx, raw_name, display_name, term_data = payload
        xs = term_data.get("names") or []
        ys = term_data.get("scores") or []
        if len(ys) == 0:
            logger.debug("Skipping term %s (no scores).", raw_name)
            return None

        categorical = not all(_is_number(x) for x in xs)
        feature_slug = _safe_filename(display_name)
        feature_dir = term_dir / feature_slug
        feature_dir.mkdir(exist_ok=True)

        write_json(
            feature_dir / "explanation.json",
            {
                "term_index": idx,
                "raw_name": raw_name,
                "display_name": display_name,
                "data": term_data,
                "plot_type": "categorical" if categorical else "continuous",
            },
        )

        _export_plotly_xy(
            xs,
            ys,
            feature_dir / "partial_dependence",
            f"{display_name} — EBM term",
            x_label=display_name,
            y_label="Contribution (logit)",
            categorical=categorical,
            density=term_data.get("density"),
        )
        return {
            "display_name": display_name,
            "raw_name": raw_name,
            "slug": feature_slug,
            "plot_html": f"terms/{feature_slug}/partial_dependence.html",
        }

    cpu_total = os.cpu_count() or 4
    max_workers = max(1, min(8, cpu_total - 2))
    timeout_seconds = 90
    max_retries = 2
    in_flight: Dict[concurrent.futures.Future, Dict[str, object]] = {}
    total_terms = len(term_payloads)
    completed = 0

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        # Kick off initial submissions
        for payload in term_payloads:
            fut = executor.submit(_export_term, payload)
            in_flight[fut] = {"payload": payload, "start": time.time(), "retries": 0}

        while in_flight:
            done, _ = concurrent.futures.wait(
                in_flight.keys(), timeout=5, return_when=concurrent.futures.FIRST_COMPLETED
            )

            # Check for completed futures
            for fut in done:
                meta = in_flight.pop(fut, None)
                try:
                    entry = fut.result()
                    if entry:
                        index_terms.append(entry)
                except Exception as exc:
                    payload = meta["payload"] if meta else None
                    logger.warning("EBM term export failed (%s); skipping. Error: %s", payload, exc)
                completed += 1
                if completed % 10 == 0 or completed == total_terms:
                    logger.info("EBM XAI exports: %s/%s terms complete", completed, total_terms)

            # Check for timeouts on remaining futures
            now = time.time()
            to_resubmit = []
            for fut, meta in list(in_flight.items()):
                start = meta["start"]
                retries = meta["retries"]
                if now - start > timeout_seconds:
                    payload = meta["payload"]
                    fut.cancel()
                    in_flight.pop(fut, None)
                    if retries < max_retries:
                        logger.warning(
                            "EBM term export timed out (attempt %s/%s); retrying: %s",
                            retries + 1,
                            max_retries,
                            payload,
                        )
                        to_resubmit.append((payload, retries + 1))
                    else:
                        logger.error("EBM term export exceeded retries; skipping: %s", payload)

            for payload, new_retry in to_resubmit:
                new_fut = executor.submit(_export_term, payload)
                in_flight[new_fut] = {"payload": payload, "start": time.time(), "retries": new_retry}

    except KeyboardInterrupt:
        logger.warning("Received KeyboardInterrupt; cancelling remaining term exports.")
        for fut in in_flight:
            fut.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        # Do not wait on worker joins to avoid hangs; cancel any remaining futures.
        for fut in in_flight:
            fut.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    _write_xai_index(
        xai_dir,
        global_section={
            "Global importances (Plotly)": "global_importances_plotly.html",
            "Global importances (PNG)": "global_importances.png",
            "Interaction importances (Plotly)": "interaction_importances_plotly.html",
            "Local contributions (PNG)": "local_contributions.png",
            "Local explanations (JSON)": "local_explanations.json",
        },
        term_entries=index_terms,
    )

    readme = xai_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# EBM XAI artifacts",
                "",
                "Local explanations in this directory are aligned with the test predictions.",
                "Raw logits come from the model's decision_function and are mapped to probabilities",
                "with the sigmoid link (p = 1 / (1 + exp(-logit))).",
                "Calibrated probabilities use logistic recalibration fitted on out-of-fold predictions:",
                f"p_calibrated = sigmoid({calibration_model.intercept:.6f} + {calibration_model.slope:.6f} * logit_raw)",
                f"(probabilities are clipped to [{calibration_model.eps}, 1 - {calibration_model.eps}]).",
                "Decisions in local_attributions.csv compare calibrated probabilities to the saved threshold",
                f"(threshold = {threshold:.6f}).",
            ]
        ),
        encoding="utf-8",
    )


def train_evaluate(
    outcome,
    branch,
    feature_set,
    smoke_test=False,
    model_type="xgboost",
    legacy_imputation=False,
    export_ebm_explanations_flag: bool = False,
    local_sample_size: int = 100,
    validation_scheme: str = "nested_cv",
    outer_folds: int = 5,
    inner_folds: int = 5,
    repeats: int = 1,
    max_workers: int = 4,
    threads_per_model: int = 8,
    resume: bool = False,
    save_final_refit: bool = False,
    n_trials: int = 100,
):
    logger.info(
        "Starting Training/Evaluation for Outcome: %s, Branch: %s, Feature Set: %s, Model: %s, Validation: %s",
        outcome,
        branch,
        feature_set,
        model_type,
        validation_scheme,
    )

    if validation_scheme not in {"nested_cv", "holdout"}:
        raise ValueError(f"Unsupported validation_scheme: {validation_scheme}")

    output_dir = utils.RESULTS_DIR / "models" / model_type / outcome / branch / feature_set
    predictions_dir = output_dir / "predictions"
    artifacts_dir = output_dir / "artifacts"
    folds_dir = artifacts_dir / "folds"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = utils.load_data(branch)
    data_file = utils.FULL_FEATURES_FILE if branch == "non_windowed" else utils.WINDOWED_FEATURES_FILE
    dataset_hash = compute_file_hash(data_file)
    preserve_nan = not legacy_imputation
    base_random_state = 42
    working_df, X, y, caseids, groups = select_modeling_dataset(df, outcome, feature_set)

    if smoke_test:
        logger.info("SMOKE TEST: reducing trials/folds/workers for %s.", validation_scheme)
        n_trials = min(n_trials, 2)
        outer_folds = min(outer_folds, 3)
        inner_folds = min(inner_folds, 3)
        max_workers = 1
        threads_per_model = min(threads_per_model, 2)

    if repeats > 1:
        raise ValueError(
            "Repeated nested CV is not implemented correctly in reporting yet. "
            "Use repeats=1 for the current supported workflow."
        )

    available_cpus = max(1, os.cpu_count() or 1)
    if threads_per_model < 1:
        raise ValueError("threads_per_model must be at least 1.")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if threads_per_model > available_cpus:
        logger.warning(
            "Reducing threads_per_model from %s to %s to fit the available CPU budget.",
            threads_per_model,
            available_cpus,
        )
        threads_per_model = available_cpus
    if validation_scheme == "nested_cv":
        max_feasible_workers = max(1, available_cpus // max(1, threads_per_model))
        if max_workers > max_feasible_workers:
            logger.warning(
                "Reducing max_workers from %s to %s to avoid oversubscribing %s CPUs "
                "(threads_per_model=%s).",
                max_workers,
                max_feasible_workers,
                available_cpus,
                threads_per_model,
            )
            max_workers = max_feasible_workers

    model_label = "XGBoost" if model_type == "xgboost" else "EBM"
    pipeline_name = f"{PIPELINE_NAME}_{validation_scheme}"
    validation_fingerprint = build_validation_fingerprint(
        dataset_hash=dataset_hash,
        validation_scheme=validation_scheme,
        model_type=model_type,
        outcome=outcome,
        branch=branch,
        feature_set=feature_set,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        repeats=repeats,
        n_trials=n_trials,
        threads_per_model=threads_per_model,
        preserve_nan=preserve_nan,
        base_random_state=base_random_state,
    )

    def _save_root_artifacts(
        train_predictions: pd.DataFrame,
        test_predictions: pd.DataFrame,
        *,
        calibration_payload: Dict[str, object],
        threshold_payload: Dict[str, object],
        metadata_payload: Dict[str, object],
        train_allow_duplicate_caseids: bool = False,
        test_allow_duplicate_caseids: bool = False,
        allow_varying_thresholds: bool = False,
    ) -> None:
        write_prediction_files(
            predictions_dir,
            train_predictions,
            test_predictions,
            logger,
            train_allow_duplicate_caseids=train_allow_duplicate_caseids,
            test_allow_duplicate_caseids=test_allow_duplicate_caseids,
            allow_varying_thresholds=allow_varying_thresholds,
        )
        write_json(artifacts_dir / "calibration.json", calibration_payload)
        write_json(artifacts_dir / "threshold.json", threshold_payload)
        write_json(artifacts_dir / "metadata.json", metadata_payload)
        logger.info("Saved calibration/threshold/metadata artifacts to %s", artifacts_dir)

    def _save_holdout_outputs(result) -> None:
        train_predictions = result.train_predictions.reset_index(drop=True)
        test_predictions = result.test_predictions.reset_index(drop=True)
        calibration_payload = {
            "intercept": result.calibration_model.intercept,
            "slope": result.calibration_model.slope,
            "eps": result.calibration_model.eps,
            "validation_scheme": validation_scheme,
        }
        threshold_payload = {
            **result.threshold_payload,
            "validation_scheme": validation_scheme,
        }
        metadata_payload = {
            **result.metadata_payload,
            "dataset_hash": dataset_hash,
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "model_type": model_type,
            "validation_scheme": validation_scheme,
        }
        _save_root_artifacts(
            train_predictions,
            test_predictions,
            calibration_payload=calibration_payload,
            threshold_payload=threshold_payload,
            metadata_payload=metadata_payload,
            train_allow_duplicate_caseids=False,
            test_allow_duplicate_caseids=False,
            allow_varying_thresholds=False,
        )
        model_path = save_model_bundle(
            output_dir,
            model_type=model_type,
            model=result.final_model,
            preprocessor=result.final_preprocessor,
        )
        logger.info("Saved model bundle to %s", model_path)
        if not smoke_test and model_type == "xgboost" and result.transformed_test_features is not None:
            save_shap_plots(result.final_model, result.transformed_test_features, output_dir)
            logger.info("Saved SHAP plots.")
        elif model_type != "xgboost":
            logger.info("Skipping SHAP generation for model type %s", model_type)

        if not smoke_test and model_type == "ebm" and export_ebm_explanations_flag:
            export_ebm_explanations(
                result.final_model,
                artifacts_dir,
                base_random_state,
                local_sample_size,
                X_local=result.transformed_test_features,
                y_local=test_predictions["y_true"],
                caseids=test_predictions["caseid"],
                raw_probabilities=test_predictions["y_prob_raw"],
                raw_logits=result.raw_test_logits,
                calibrated_probabilities=test_predictions["y_prob_calibrated"],
                threshold=result.threshold_payload["threshold"],
                calibration_model=result.calibration_model,
            )
        elif model_type == "ebm":
            logger.info("Skipping EBM explanation export (disabled or smoke test)")

    def _save_final_refit_bundle() -> None:
        final_refit_dir = artifacts_dir / "final_refit"
        final_refit_dir.mkdir(parents=True, exist_ok=True)
        best_params, best_score, actual_inner = tune_model(
            X,
            y,
            groups,
            model_type=model_type,
            n_trials=n_trials,
            requested_splits=inner_folds,
            random_state=base_random_state,
            preserve_nan=preserve_nan,
            threads_per_model=threads_per_model,
        )
        oof_predictions, fold_indices, actual_calibration = generate_cross_fitted_predictions(
            X,
            y,
            groups,
            model_type=model_type,
            best_params=best_params,
            requested_splits=inner_folds,
            random_state=base_random_state,
            preserve_nan=preserve_nan,
            threads_per_model=threads_per_model,
        )
        from model_creation.postprocessing import apply_logistic_recalibration, fit_logistic_recalibration, find_youden_j_threshold

        recalibration_model = fit_logistic_recalibration(y.values, oof_predictions)
        calibrated_oof = apply_logistic_recalibration(oof_predictions, recalibration_model)
        threshold, youden_j, sensitivity, specificity = find_youden_j_threshold(y.values, calibrated_oof)
        effective_refit_params = get_effective_refit_params(model_type, best_params)
        final_preprocessor, final_model, _ = fit_final_model(
            X,
            y,
            model_type=model_type,
            best_params=best_params,
            random_state=base_random_state,
            preserve_nan=preserve_nan,
            threads_per_model=threads_per_model,
        )
        model_path = save_model_bundle(
            final_refit_dir,
            model_type=model_type,
            model=final_model,
            preprocessor=final_preprocessor,
        )
        refit_oof = pd.DataFrame(
            {
                "caseid": caseids.values,
                "y_true": y.values,
                "y_prob_raw": oof_predictions,
                "y_prob_calibrated": calibrated_oof,
                "threshold": threshold,
                "y_pred_label": (calibrated_oof >= threshold).astype(int),
                "fold": fold_indices,
                "is_oof": True,
                "outcome": outcome,
                "branch": branch,
                "feature_set": feature_set,
                "model_name": model_label,
                "pipeline": f"{pipeline_name}_final_refit",
                "validation_scheme": "final_refit",
                "outer_fold_id": -1,
                "repeat_id": 0,
            }
        )
        if groups is not None:
            refit_oof["subjectid"] = groups.values
        refit_oof.to_csv(final_refit_dir / "train_oof.csv", index=False)
        write_json(
            final_refit_dir / "calibration.json",
            {
                "intercept": recalibration_model.intercept,
                "slope": recalibration_model.slope,
                "eps": recalibration_model.eps,
            },
        )
        write_json(
            final_refit_dir / "threshold.json",
            {
                "threshold": threshold,
                "youden_j": youden_j,
                "sensitivity": sensitivity,
                "specificity": specificity,
            },
        )
        write_json(
            final_refit_dir / "metadata.json",
            {
                "dataset_hash": dataset_hash,
                "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
                "source_validation_fingerprint": validation_fingerprint,
                "best_score": best_score,
                "best_params": best_params,
                "effective_refit_params": effective_refit_params,
                "refit_param_overrides": get_refit_param_overrides(model_type, best_params),
                "actual_inner_folds": actual_inner,
                "actual_calibration_folds": actual_calibration,
                "model_path": str(model_path.name),
            },
        )

    if validation_scheme == "holdout":
        train_mask = working_df["split_group"] == "train" if "split_group" in working_df.columns else None
        test_mask = working_df["split_group"] == "test" if "split_group" in working_df.columns else None
        if train_mask is None or test_mask is None:
            if groups is None:
                raise ValueError("Holdout validation requires patient groups or an existing split_group column.")
            split_train_idx, split_test_idx = utils.select_patient_level_holdout_positions(
                y, groups, random_state=base_random_state
            )
        else:
            split_train_idx = np.flatnonzero(train_mask.to_numpy())
            split_test_idx = np.flatnonzero(test_mask.to_numpy())
        result = run_outer_split(
            df=working_df,
            X=X,
            y=y,
            caseids=caseids,
            groups=groups,
            train_idx=split_train_idx,
            test_idx=split_test_idx,
            outcome=outcome,
            branch=branch,
            feature_set=feature_set,
            model_type=model_type,
            model_label=model_label,
            pipeline_name=pipeline_name,
            validation_scheme=validation_scheme,
            n_trials=n_trials,
            inner_folds=inner_folds,
            random_state=base_random_state,
            preserve_nan=preserve_nan,
            threads_per_model=threads_per_model,
            repeat_id=0,
            outer_fold_id=0,
            return_fitted_model=True,
        )
        params_dir = utils.RESULTS_DIR / "params" / model_type / outcome / branch
        params_dir.mkdir(parents=True, exist_ok=True)
        with (params_dir / f"{feature_set}.json").open("w") as fp:
            json.dump(result.best_params, fp, indent=4)
        _save_holdout_outputs(result)
        final_refit_saved = False
        if save_final_refit:
            logger.info("Running optional final refit on the full dataset.")
            _save_final_refit_bundle()
            final_refit_saved = True
        write_validation_manifest(
            artifacts_dir,
            validation_fingerprint=validation_fingerprint,
            summary_metadata={
                "outer_folds": 1,
                "inner_folds": int(result.metadata_payload["actual_inner_folds"]),
                "repeats": 1,
                "threads_per_model": threads_per_model,
                "n_trials": n_trials,
                "resume": False,
                "final_refit_saved": final_refit_saved,
            },
        )
        logger.info("Training and evaluation complete.")
        return

    manifest_entries: List[Dict[str, object]] = []
    outer_jobs: Dict[Tuple[int, int], Dict[str, object]] = {}
    actual_outer_folds = 0

    for repeat_id in range(repeats):
        repeat_seed = base_random_state + repeat_id
        split_list, resolved_outer_folds = build_cv_splits(
            X,
            y,
            groups,
            outer_folds,
            repeat_seed,
        )
        actual_outer_folds = max(actual_outer_folds, resolved_outer_folds)
        for outer_fold_id, (train_idx, test_idx) in enumerate(split_list):
            fold_dir = folds_dir / f"repeat_{repeat_id:02d}" / f"outer_{outer_fold_id:02d}"
            outer_jobs[(repeat_id, outer_fold_id)] = {
                "train_idx": train_idx,
                "test_idx": test_idx,
                "fold_dir": fold_dir,
                "random_state": repeat_seed + outer_fold_id,
            }

    results_by_key: Dict[Tuple[int, int], object] = {}
    if resume:
        for key, meta in outer_jobs.items():
            if is_outer_fold_checkpoint_complete(meta["fold_dir"]):
                checkpoint_matches, mismatch_reason = checkpoint_matches_validation_fingerprint(
                    meta["fold_dir"],
                    validation_fingerprint,
                )
                if not checkpoint_matches:
                    logger.warning(
                        "Ignoring checkpoint for repeat=%s outer=%s: %s",
                        key[0],
                        key[1],
                        mismatch_reason,
                    )
                    continue
                logger.info("Resuming from completed outer fold %s.", key)
                results_by_key[key] = load_outer_fold_checkpoint(meta["fold_dir"])
                manifest_entries.append(
                    {
                        "repeat_id": key[0],
                        "outer_fold_id": key[1],
                        "status": "resumed",
                        "path": str(meta["fold_dir"]),
                    }
                )

    pending = [(key, meta) for key, meta in outer_jobs.items() if key not in results_by_key]
    logger.info(
        "Running nested CV with %s outer jobs (%s resumed, %s pending, max_workers=%s, threads_per_model=%s).",
        len(outer_jobs),
        len(results_by_key),
        len(pending),
        max_workers,
        threads_per_model,
    )

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    run_outer_split,
                    df=working_df,
                    X=X,
                    y=y,
                    caseids=caseids,
                    groups=groups,
                    train_idx=meta["train_idx"],
                    test_idx=meta["test_idx"],
                    outcome=outcome,
                    branch=branch,
                    feature_set=feature_set,
                    model_type=model_type,
                    model_label=model_label,
                    pipeline_name=pipeline_name,
                    validation_scheme=validation_scheme,
                    n_trials=n_trials,
                    inner_folds=inner_folds,
                    random_state=meta["random_state"],
                    preserve_nan=preserve_nan,
                    threads_per_model=threads_per_model,
                    repeat_id=key[0],
                    outer_fold_id=key[1],
                ): (key, meta)
                for key, meta in pending
            }
            for future in concurrent.futures.as_completed(future_map):
                key, meta = future_map[future]
                result = future.result()
                save_outer_fold_checkpoint(
                    meta["fold_dir"],
                    result,
                    validation_fingerprint=validation_fingerprint,
                )
                results_by_key[key] = result
                manifest_entries.append(
                    {
                        "repeat_id": key[0],
                        "outer_fold_id": key[1],
                        "status": "completed",
                        "path": str(meta["fold_dir"]),
                        "best_score": result.best_score,
                    }
                )
                logger.info("Completed nested outer fold repeat=%s outer=%s.", key[0], key[1])

    ordered_keys = sorted(results_by_key)
    train_predictions = pd.concat(
        [results_by_key[key].train_predictions for key in ordered_keys], ignore_index=True
    )
    test_predictions = pd.concat(
        [results_by_key[key].test_predictions for key in ordered_keys], ignore_index=True
    )
    train_predictions = train_predictions.sort_values(["repeat_id", "outer_fold_id", "caseid"]).reset_index(drop=True)
    test_predictions = test_predictions.sort_values(["repeat_id", "outer_fold_id", "caseid"]).reset_index(drop=True)
    _assert_nested_test_prediction_coverage(test_predictions, caseids)

    overall_positive = int((test_predictions["y_true"] == 1).sum())
    overall_negative = int((test_predictions["y_true"] == 0).sum())
    _save_root_artifacts(
        train_predictions,
        test_predictions,
        calibration_payload={
            "validation_scheme": validation_scheme,
            "per_fold": True,
            "fold_artifacts_dir": "folds",
            "fold_count": len(ordered_keys),
        },
        threshold_payload={
            "validation_scheme": validation_scheme,
            "per_fold": True,
            "threshold_source": "rowwise_prediction_metadata",
            "fold_count": len(ordered_keys),
        },
        metadata_payload={
            "seed": base_random_state,
            "outer_folds": actual_outer_folds,
            "inner_folds": inner_folds,
            "repeats": repeats,
            "dataset_hash": dataset_hash,
            "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
            "model_type": model_type,
            "validation_scheme": validation_scheme,
            "counts": {
                "train": {
                    "total": int(len(train_predictions)),
                    "positive": int((train_predictions["y_true"] == 1).sum()),
                    "negative": int((train_predictions["y_true"] == 0).sum()),
                },
                "test": {
                    "total": int(len(test_predictions)),
                    "positive": overall_positive,
                    "negative": overall_negative,
                },
            },
        },
        train_allow_duplicate_caseids=train_predictions["caseid"].duplicated().any(),
        test_allow_duplicate_caseids=False,
        allow_varying_thresholds=True,
    )
    final_refit_saved = False
    if save_final_refit:
        logger.info("Running optional final refit on the full dataset.")
        _save_final_refit_bundle()
        final_refit_saved = True

    write_validation_manifest(
        artifacts_dir,
        validation_fingerprint=validation_fingerprint,
        summary_metadata={
            "outer_folds": actual_outer_folds,
            "inner_folds": inner_folds,
            "repeats": repeats,
            "threads_per_model": threads_per_model,
            "max_workers": max_workers,
            "n_trials": n_trials,
            "resume": resume,
            "fold_manifest": "folds_manifest.json",
            "final_refit_saved": final_refit_saved,
        },
    )
    write_json(artifacts_dir / "folds_manifest.json", {"folds": manifest_entries})

    logger.info("Training and evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate AKI Prediction Models")
    parser.add_argument("--outcome", type=str, required=True, help="Target outcome name")
    parser.add_argument("--branch", type=str, required=True, choices=['windowed', 'non_windowed'], help="Data branch")
    parser.add_argument("--feature_set", type=str, required=True, help="Feature set name")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["xgboost", "ebm"],
        default="xgboost",
        help="Model type to train and evaluate",
    )
    parser.add_argument(
        "--legacy_imputation",
        action="store_true",
        help="Apply legacy imputation instead of preserving NaNs",
    )
    parser.add_argument(
        "--export_ebm_explanations",
        action="store_true",
        help="Export EBM global and local explanations after training",
    )
    parser.add_argument(
        "--local_sample_size",
        type=int,
        default=100,
        help="Number of training samples to include in local EBM explanations",
    )
    parser.add_argument(
        "--validation-scheme",
        type=str,
        choices=["nested_cv", "holdout"],
        default="nested_cv",
        help="Validation scheme to run (default: nested_cv).",
    )
    parser.add_argument("--outer-folds", type=int, default=5, help="Number of outer folds for nested CV.")
    parser.add_argument("--inner-folds", type=int, default=5, help="Number of inner folds for HPO/calibration.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of outer-CV repetitions. Values greater than 1 are currently unsupported.",
    )
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel outer-fold workers.")
    parser.add_argument("--threads-per-model", type=int, default=8, help="Threads allocated to each fitted model.")
    parser.add_argument("--resume", action="store_true", help="Resume from completed outer-fold checkpoints.")
    parser.add_argument(
        "--save-final-refit",
        action="store_true",
        help="After validation, tune/refit on all available data and save the final artifact bundle.",
    )
    parser.add_argument("--n-trials", type=int, default=100, help="Optuna trials per tuning run.")

    args = parser.parse_args()

    train_evaluate(
        args.outcome,
        args.branch,
        args.feature_set,
        args.smoke_test,
        args.model_type,
        args.legacy_imputation,
        args.export_ebm_explanations,
        args.local_sample_size,
        args.validation_scheme,
        args.outer_folds,
        args.inner_folds,
        args.repeats,
        args.max_workers,
        args.threads_per_model,
        args.resume,
        args.save_final_refit,
        args.n_trials,
    )
