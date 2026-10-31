"""Paired DeLong calculations, restricted to verified independent holdouts.

The tie-aware placement formula follows DeLong et al. (1988) and agrees with
the Sun/Xu midrank implementation. Ordinary DeLong inference does not account
for patient clusters or shared-training dependence in pooled cross-validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _validated_arrays(y_true, scores_reference, scores_candidate):
    labels = np.asarray(y_true)
    try:
        reference = np.asarray(scores_reference, dtype=float)
        candidate = np.asarray(scores_candidate, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("DeLong scores must be finite numeric vectors") from exc
    if any(values.ndim != 1 for values in (labels, reference, candidate)):
        raise ValueError("DeLong requires one-dimensional vectors")
    if not len(labels) or len(labels) != len(reference) or len(labels) != len(candidate):
        raise ValueError("DeLong requires nonempty vectors of equal length")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("DeLong labels must be binary 0/1")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("DeLong scores must be finite numeric vectors")
    return labels.astype(int), reference, candidate


def _placements(labels: np.ndarray, scores: np.ndarray):
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan"), np.array([]), np.array([])
    sorted_negative, sorted_positive = np.sort(negative), np.sort(positive)
    left = np.searchsorted(sorted_negative, positive, side="left")
    right = np.searchsorted(sorted_negative, positive, side="right")
    positive_placements = (left + 0.5 * (right - left)) / len(negative)
    left = np.searchsorted(sorted_positive, negative, side="left")
    right = np.searchsorted(sorted_positive, negative, side="right")
    negative_placements = (len(positive) - right + 0.5 * (right - left)) / len(positive)
    return float(positive_placements.mean()), positive_placements, negative_placements


def _result(labels, reference_auc, candidate_auc, *, status, reason):
    return {
        "reference_auc": reference_auc, "candidate_auc": candidate_auc,
        "delta": candidate_auc - reference_auc,
        "variance": float("nan"), "p_value": float("nan"),
        "log_p_value": float("nan"),
        "n_operations": len(labels), "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "method": "ordinary_paired_delong", "status": status, "reason": reason,
    }


def paired_delong(y_true, scores_reference, scores_candidate) -> dict:
    """Return candidate-minus-reference AUROC inference for paired scores.

    ``log_p_value`` is the natural logarithm of the two-sided p-value. It stays
    finite in tails where the ordinary p-value underflows. This numerical
    function assumes independent observations; use ``gated_delong`` for saved
    study artifacts. Invalid inputs raise, while zero variance is explicit.
    """
    labels, reference, candidate = _validated_arrays(y_true, scores_reference, scores_candidate)
    n_positive, n_negative = np.sum(labels == 1), np.sum(labels == 0)
    if min(n_positive, n_negative) < 2:
        raise ValueError("DeLong covariance requires at least two observations per class")
    ref_auc, ref_positive, ref_negative = _placements(labels, reference)
    cand_auc, cand_positive, cand_negative = _placements(labels, candidate)
    result = _result(labels, ref_auc, cand_auc, status="ok", reason="independent-observation numerical calculation")
    # Differences avoid cancellation from subtracting almost equal covariance
    # terms and give exactly zero for identical structural placements.
    variance = float(np.var(cand_positive - ref_positive, ddof=1) / n_positive
                     + np.var(cand_negative - ref_negative, ddof=1) / n_negative)
    result["variance"] = variance
    if variance <= 0 or not np.isfinite(variance):
        result.update(status="degenerate", reason="nonpositive_or_nonfinite_auc_difference_variance")
        return result
    z_score = abs(result["delta"]) / np.sqrt(variance)
    result["p_value"] = float(min(1.0, 2 * norm.sf(z_score)))
    result["log_p_value"] = float(min(0.0, np.log(2.0) + norm.logsf(z_score)))
    return result


def _canonical_frame(frame: pd.DataFrame, *, require_scores: bool):
    frame = frame.copy()
    if "subjectid" not in frame and "subject_id" in frame:
        frame = frame.rename(columns={"subject_id": "subjectid"})
    required = ["caseid", "subjectid", "y_true"] + (["y_prob_calibrated"] if require_scores else [])
    if any(column not in frame for column in required):
        raise ValueError(f"DeLong requires columns {required}")
    if frame[required].isna().any().any() or frame.caseid.duplicated().any():
        raise ValueError("DeLong requires unique cases and nonmissing identity/outcomes/scores")
    if not np.isin(frame.y_true, [0, 1]).all():
        raise ValueError("DeLong labels must be binary 0/1")
    return frame.sort_values("caseid", kind="stable").reset_index(drop=True)


def _verified_holdout(ps, frame):
    """Check saved train/test identity without loading or fitting an estimator."""
    prediction_path = Path(ps.path)
    metadata_path = prediction_path.parent.parent / "artifacts" / "metadata.json"
    training_path = prediction_path.parent / "train_oof.csv"
    if not all(path.is_file() for path in (prediction_path, metadata_path, training_path)):
        return None, "missing_holdout_provenance"
    try:
        metadata = json.loads(metadata_path.read_text())
        fingerprint = metadata.get("validation_fingerprint", {})
        if (metadata.get("validation_scheme") != "holdout" or fingerprint.get("validation_scheme") != "holdout"
                or fingerprint.get("cohort_scope") != ps.cohort_scope
                or any(not fingerprint.get(key) for key in ("selected_case_hash", "fold_assignment_hash", "dataset_hash"))):
            return None, "incomplete_holdout_fingerprint"
        saved = _canonical_frame(pd.read_csv(prediction_path), require_scores=True)
        identity = ["caseid", "subjectid", "y_true", "outer_fold_id", "repeat_id"]
        if (any(column not in saved for column in identity) or len(saved) != len(frame)
                or not np.array_equal(saved[identity].to_numpy(), frame[identity].to_numpy())
                or not np.allclose(saved.y_prob_calibrated, frame.y_prob_calibrated, rtol=0, atol=1e-12)):
            return None, "saved_test_predictions_do_not_match"
        training = _canonical_frame(pd.read_csv(training_path), require_scores=False)
        if set(training.caseid) & set(frame.caseid) or set(training.subjectid) & set(frame.subjectid):
            return None, "training_test_overlap"
        from data_preparation.cohort_scopes import selected_case_hash
        combined = pd.concat([training, frame], ignore_index=True)
        if selected_case_hash(combined) != fingerprint["selected_case_hash"]:
            return None, "selected_case_hash_mismatch"
    except (ValueError, TypeError, OSError, KeyError, AttributeError):
        return None, "invalid_holdout_provenance"
    return (training, fingerprint), "verified_independent_holdout"


def gated_delong(reference, candidate) -> dict:
    """Apply ordinary DeLong only to verified one-operation-per-patient holdouts.

    Arguments expose ``df``, ``path``, ``cohort_scope``, ``outcome``, ``branch``
    and ``validation_scheme`` like ``PredictionSet``. Mispaired or invalid data
    raises even if the design would otherwise be unsupported. A valid but
    unsupported design returns a missing p-value and an explicit reason.
    """
    ref = _canonical_frame(reference.df, require_scores=True)
    cand = _canonical_frame(candidate.df, require_scores=True)
    for attribute in ("cohort_scope", "outcome", "branch", "validation_scheme"):
        if getattr(reference, attribute, None) != getattr(candidate, attribute, None):
            raise ValueError(f"DeLong comparison differs in {attribute}")
    columns = ["caseid", "subjectid", "y_true"]
    for column in ("outer_fold_id", "repeat_id", "fold"):
        if column in ref or column in cand:
            if column not in ref or column not in cand or ref[column].isna().any() or cand[column].isna().any():
                raise ValueError(f"DeLong requires matching nonmissing {column}")
            columns.append(column)
    if not np.array_equal(ref[columns].to_numpy(), cand[columns].to_numpy()):
        raise ValueError("DeLong case/patient/truth/fold identities do not match")
    labels, ref_scores, cand_scores = _validated_arrays(ref.y_true, ref.y_prob_calibrated, cand.y_prob_calibrated)
    ref_auc = _placements(labels, ref_scores)[0]
    cand_auc = _placements(labels, cand_scores)[0]
    result = _result(labels, ref_auc, cand_auc, status="unsupported_design", reason="")
    result["n_patients"] = int(ref.subjectid.nunique())
    if reference.validation_scheme != "holdout":
        result["reason"] = "pooled_cv_or_unverified_evaluation_design"
    elif ref.subjectid.duplicated().any():
        result["reason"] = "repeated_patient_operations"
    elif any(column not in ref or ref[column].nunique() != 1 for column in ("outer_fold_id", "repeat_id")):
        result["reason"] = "missing_or_multiple_holdout_fold_identities"
    elif min(result["n_positive"], result["n_negative"]) < 2:
        result["reason"] = "insufficient_class_support"
    else:
        verified_ref, reason = _verified_holdout(reference, ref)
        verified_cand, candidate_reason = _verified_holdout(candidate, cand)
        if verified_ref is None or verified_cand is None:
            result["reason"] = reason if verified_ref is None else candidate_reason
        else:
            training_ref, fingerprint_ref = verified_ref
            training_cand, fingerprint_cand = verified_cand
            if (not np.array_equal(training_ref[["caseid", "subjectid", "y_true"]].to_numpy(), training_cand[["caseid", "subjectid", "y_true"]].to_numpy())
                    or any(fingerprint_ref[key] != fingerprint_cand[key] for key in ("selected_case_hash", "fold_assignment_hash", "dataset_hash"))):
                result["reason"] = "holdout_training_or_fold_provenance_mismatch"
            else:
                result.update(paired_delong(labels, ref_scores, cand_scores))
                if result["status"] == "ok":
                    result["reason"] = "verified_independent_holdout_conditional_on_fitted_models"
    return result


def add_holm_adjustment(frame: pd.DataFrame, group_columns=("cohort_scope", "validation_scheme", "outcome", "branch")) -> pd.DataFrame:
    """Holm-adjust unique eligible comparisons, retaining stable natural log p."""
    result = frame.copy().reset_index(drop=True)
    result["p_value_holm"] = float("nan")
    result["log_p_value_holm"] = float("nan")
    result["multiplicity_method"] = "Holm within cohort/outcome/branch"
    if result.empty:
        return result
    if "comparison_id" in result and result.comparison_id.duplicated().any():
        raise ValueError("Holm adjustment requires unique comparison IDs")
    columns = [column for column in group_columns if column in result]
    groups = result.groupby(columns, dropna=False).groups.values() if columns else [result.index]
    for indices in groups:
        eligible = result.loc[indices].loc[lambda rows: rows.status.eq("ok")]
        log_values = pd.to_numeric(eligible.get("log_p_value", pd.Series(np.nan, index=eligible.index)), errors="coerce")
        raw_values = pd.to_numeric(eligible.get("p_value", pd.Series(np.nan, index=eligible.index)), errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            log_values = log_values.where(log_values.notna(), np.log(raw_values))
        if ((log_values > 0) | (raw_values < 0) | (raw_values > 1)).any():
            raise ValueError("Holm adjustment requires probabilities in [0,1]")
        log_values = log_values.dropna().sort_values(kind="stable")
        running = -np.inf
        for rank, (index, log_value) in enumerate(log_values.items()):
            running = min(0.0, max(running, float(log_value) + np.log(len(log_values) - rank)))
            result.loc[index, "log_p_value_holm"] = running
            result.loc[index, "p_value_holm"] = float(np.exp(running))
    return result
