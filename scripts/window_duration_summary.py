"""Summarize an outcome-blind Catch24 window-origin pilot; never fit models.

Input aggregates have axes patient, channel, length, origin, reducer, feature.
The engineering gates are specified in window-duration-pilot.md. Bootstrap
weights resample patients jointly across every other axis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REDUCERS = ("mean", "sd", "min", "max")


def _prepare(aggregates):
    values = np.asarray(aggregates, dtype=np.float64)
    if (values.ndim != 6 or any(size == 0 for size in values.shape)
            or values.shape[3:] != (4, 4, 24)):
        raise ValueError("Expected nonempty patient/channel/length/4-origin/4-reducer/24-feature axes")
    complete = np.isfinite(values).all(axis=3)
    with np.errstate(invalid="ignore", over="ignore"):
        medians = np.median(values, axis=3)
        ranges = np.max(values, axis=3) - np.min(values, axis=3)
    medians = np.where(complete, medians, np.nan)
    ranges = np.where(complete, ranges, np.nan)
    prepared = {"n_patients": len(values), "complete": complete,
                "available": complete.all(axis=3)}
    # Sort once. A patient bootstrap changes only integer frequency weights,
    # so weighted order statistics reproduce sorting every resampled matrix.
    for key, array in (("medians", medians), ("ranges", ranges)):
        array = np.moveaxis(array, 0, -1)
        order = np.argsort(array, axis=-1)
        prepared[key] = (np.take_along_axis(array, order, axis=-1), order)
    return prepared


def _weighted_quantiles(sorted_values, order, weights, probabilities):
    ordered_weights = weights[order] * np.isfinite(sorted_values)
    cumulative = ordered_weights.cumsum(axis=-1)
    count = cumulative[..., -1]
    quantiles = []
    for probability in probabilities:
        rank = np.maximum(count - 1, 0) * probability
        lower, upper = np.floor(rank).astype(int), np.ceil(rank).astype(int)
        left = np.argmax(cumulative > lower[..., None], axis=-1)
        right = np.argmax(cumulative > upper[..., None], axis=-1)
        lo = np.take_along_axis(sorted_values, left[..., None], axis=-1)[..., 0]
        hi = np.take_along_axis(sorted_values, right[..., None], axis=-1)[..., 0]
        with np.errstate(invalid="ignore", over="ignore"):
            value = lo + (hi - lo) * (rank - lower)
        quantiles.append(np.where(count > 0, value, np.nan))
    max_abs = np.max(np.where(ordered_weights > 0, np.abs(sorted_values), 0), axis=-1)
    return quantiles, max_abs


def _score(prepared, weights):
    required = int(np.ceil(0.95 * weights.sum()))
    n_complete = np.tensordot(weights, prepared["complete"], axes=(0, 0))
    n_available = np.tensordot(weights, prepared["available"], axes=(0, 0))
    (q25, q75), max_abs = _weighted_quantiles(*prepared["medians"], weights, [0.25, 0.75])
    (p90_range,), _ = _weighted_quantiles(*prepared["ranges"], weights, [0.90])
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        scale = q75 - q25
        normalized = p90_range / scale
    guard = 100 * np.finfo(np.float64).eps * max_abs
    evaluable = ((n_complete >= required) & np.isfinite(scale) & (scale > 0)
                 & (scale > guard) & np.isfinite(p90_range) & np.isfinite(normalized))
    passed = evaluable & (normalized <= 0.25)
    availability_pass = n_available >= required
    return {"n_complete": n_complete, "n_available": n_available,
            "scale": scale, "p90_abs_range": p90_range,
            "p90_normalized_range": np.where(evaluable, normalized, np.nan),
            "evaluable": evaluable, "pass": passed,
            "availability_pass": availability_pass,
            "availability_pass_counts": availability_pass.sum(axis=-1),
            "origin_pass_counts": passed.sum(axis=-1), "required_patients": required}


def score_aggregates(aggregates):
    """Point gates; undefined values fail their output, never an entire draw."""
    prepared = _prepare(aggregates)
    return _score(prepared, np.ones(prepared["n_patients"], dtype=int))


def select_candidate(lengths, availability_pass_counts, origin_pass_counts):
    """Return the shortest passing adjacent-doubling pair or None."""
    lengths = np.asarray(lengths, dtype=int)
    availability = np.asarray(availability_pass_counts)
    origin = np.asarray(origin_pass_counts)
    if (lengths.ndim != 1 or not len(lengths) or np.any(lengths <= 0)
            or np.any(np.diff(lengths) <= 0)
            or availability.shape != origin.shape[:2]
            or availability.shape[1] != len(lengths) or origin.shape[-1] != 4):
        raise ValueError("Lengths and channel/length/reducer gate counts do not align")
    quality = (availability >= 22).all(axis=0) & (origin >= 20).all(axis=(0, 2))
    for index, length in enumerate(lengths):
        matches = np.flatnonzero(lengths == length * 2)
        if not len(matches):
            continue
        other = int(matches[0])
        if (quality[index] and quality[other]
                and np.all(origin[:, other] - origin[:, index] <= 1)
                and np.all(availability[:, other] - availability[:, index] <= 1)):
            return int(length)
    return None


def _null_nonfinite(value):
    if isinstance(value, np.ndarray):
        return _null_nonfinite(value.tolist())
    if isinstance(value, np.generic):
        return _null_nonfinite(value.item())
    if isinstance(value, dict):
        return {str(key): _null_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_null_nonfinite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize(aggregates, lengths, *, bootstrap=2000, seed=0,
              channels=None, feature_names=None):
    """Return strict-JSON-ready summary plus point metrics/availability tables."""
    if bootstrap < 0:
        raise ValueError("Bootstrap count must be nonnegative")
    prepared = _prepare(aggregates)
    n_patients = prepared["n_patients"]
    n_channels, n_lengths, _, n_features = prepared["complete"].shape[1:]
    if len(lengths) != n_lengths:
        raise ValueError("Length metadata does not match aggregate axes")
    channels = list(channels) if channels is not None else [f"channel_{i}" for i in range(n_channels)]
    feature_names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(n_features)]
    if len(channels) != n_channels or len(feature_names) != n_features:
        raise ValueError("Channel/feature metadata does not match aggregate axes")
    point = _score(prepared, np.ones(n_patients, dtype=int))
    candidate = select_candidate(lengths, point["availability_pass_counts"], point["origin_pass_counts"])
    availability_draws = np.empty((bootstrap, n_channels, n_lengths), dtype=np.int16)
    origin_draws = np.empty((bootstrap, n_channels, n_lengths, 4), dtype=np.int16)
    frequencies = {str(int(length)): 0 for length in lengths}
    frequencies["no_selection"] = 0
    rng = np.random.default_rng(seed)
    for draw in range(bootstrap):
        weights = np.bincount(rng.integers(n_patients, size=n_patients), minlength=n_patients)
        scored = _score(prepared, weights)
        availability_draws[draw] = scored["availability_pass_counts"]
        origin_draws[draw] = scored["origin_pass_counts"]
        selected = select_candidate(lengths, availability_draws[draw], origin_draws[draw])
        frequencies["no_selection" if selected is None else str(selected)] += 1
    stability = frequencies[str(candidate)] / bootstrap if bootstrap and candidate is not None else None
    rows, available_rows = [], []
    for c, channel in enumerate(channels):
        for l, length in enumerate(lengths):
            for f, feature in enumerate(feature_names):
                available_rows.append({"channel": channel, "length": int(length), "feature": feature,
                                       "n_available": int(point["n_available"][c, l, f]),
                                       "availability_fraction": point["n_available"][c, l, f] / n_patients,
                                       "pass": bool(point["availability_pass"][c, l, f])})
                for r, reducer in enumerate(REDUCERS):
                    key = (c, l, r, f)
                    rows.append({"channel": channel, "length": int(length), "reducer": reducer, "feature": feature,
                                 **{field: point[field][key] for field in
                                    ["n_complete", "scale", "p90_abs_range", "p90_normalized_range", "evaluable", "pass"]}})
    summary = {
        "axes": ["patient", "channel", "length", "origin", "reducer", "feature"],
        "shape": list(np.shape(aggregates)), "lengths": list(lengths), "channels": channels,
        "reducers": REDUCERS, "feature_names": feature_names,
        "engineering_gates": {"minimum_patient_fraction": 0.95, "required_patients": point["required_patients"],
                              "availability_descriptors_per_channel": 22, "origin_descriptors_per_channel_reducer": 20,
                              "p90_origin_range_over_iqr_maximum": 0.25, "maximum_plateau_gain_descriptors": 1,
                              "bootstrap_selection_frequency_minimum": 0.80, "quantile_method": "linear (type 7)"},
        "availability_pass_counts": point["availability_pass_counts"],
        "origin_pass_counts": point["origin_pass_counts"],
        "case_failure_counts_axes": ["patient", "channel", "length"],
        "case_unavailable_descriptor_counts": (~prepared["available"]).sum(axis=-1),
        "case_incomplete_output_counts": (~prepared["complete"]).sum(axis=(3, 4)),
        "technical_candidate_seconds": candidate,
        "decision": "stable_technical_candidate" if stability is not None and stability >= 0.80 else "inconclusive",
        "bootstrap": {"replicates": bootstrap, "seed": seed, "selection_counts": frequencies,
                      "selection_frequencies": {key: count / bootstrap if bootstrap else None for key, count in frequencies.items()},
                      "point_candidate_selection_frequency": stability,
                      "percentile_levels": [2.5, 50, 97.5],
                      "availability_pass_count_percentiles": np.percentile(availability_draws, [2.5, 50, 97.5], axis=0) if bootstrap else None,
                      "origin_pass_count_percentiles": np.percentile(origin_draws, [2.5, 50, 97.5], axis=0) if bootstrap else None},
        "interpretation": "Window-origin robustness conditional on sampled recordings; engineering thresholds do not validate physiology or outcome prediction.",
    }
    return _null_nonfinite(summary), pd.DataFrame(rows), pd.DataFrame(available_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must be fresh and absent")
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    with np.load(args.input, allow_pickle=False) as bundle:
        aggregates = bundle["aggregates"]
        summary, metrics, availability = summarize(
            aggregates, bundle["lengths"], bootstrap=args.bootstrap,
            channels=bundle["channels"].tolist(), feature_names=bundle["feature_names"].tolist())
        for field in ("patient_ids", "case_ids"):
            identifiers = bundle[field]
            if identifiers.ndim != 1 or len(identifiers) != len(aggregates) or len(np.unique(identifiers)) != len(identifiers):
                raise ValueError(f"Expected one distinct {field} per patient operation")
            summary[field] = _null_nonfinite(identifiers)
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Pilot input changed during summary")
    summary["input"] = {"path": str(args.input.resolve()), "sha256": digest}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    availability.to_csv(args.output_dir / "availability.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"{summary['decision']}: {summary['technical_candidate_seconds']} seconds; summary at {args.output_dir}")


if __name__ == "__main__":
    main()
