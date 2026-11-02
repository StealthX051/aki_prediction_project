"""Known-change Catch24 challenge; no clinical outcomes or estimator fitting.

Signals are generated at feature-input rates, without additional filtering.
Run with ``python -m scripts.window_duration_synthetic --output-dir FRESH_DIR``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from scripts.window_duration_resources import adaptive_map, PilotDrained


LENGTHS = (10, 20, 40, 80, 160, 320, 640)
RATES = (100.0, 62.5)
GENERATORS = ("sine_0p2", "sine_1", "ar1")
NOISE_LEVELS = (0.1, 0.3)
REDUCERS = ("mean", "std", "min", "max")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_stem(rate: float, generator: str, noise: float, seed: int) -> str:
    return f"rate_{rate:g}__{generator}__noise_{noise:g}__seed_{seed:02d}"


def verified_trace(output_dir: Path, stem: str, signature: str) -> dict | None:
    """Only metadata published last can declare a completed trace reusable."""
    path = output_dir / f"{stem}.json"
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    if metadata.get("protocol_sha256") != signature:
        raise ValueError(f"Synthetic trace protocol changed: {stem}")
    checksums = metadata.get("artifact_sha256", {})
    if set(checksums) != {f"{stem}.npz", f"{stem}.csv"}:
        raise ValueError(f"Synthetic trace lacks complete artifact checksums: {stem}")
    for name, digest in checksums.items():
        artifact = output_dir / name
        if not artifact.is_file() or sha256(artifact) != digest:
            raise ValueError(f"Synthetic checkpoint artifact changed: {name}")
    return {"trace": stem, "feature_rows": metadata["feature_rows"],
            "window_counts": metadata["window_counts"], "reused": True}


def write_json(path: Path, value: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    partial.replace(path)


def episode_envelope(times: np.ndarray, start: float, end: float, ramp: float) -> np.ndarray:
    """Five-second linear ramps lie inside the episode, with no edge jump."""
    if end <= start or ramp <= 0 or 2 * ramp > end - start:
        raise ValueError("Episode must contain both positive-duration ramps")
    return np.clip(np.minimum((times - start) / ramp, (end - times) / ramp), 0, 1)


def ar_process(innovations: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Unit stationary variance at constant phi; changing phi keeps state continuity."""
    if len(innovations) != len(phi) or np.any(np.abs(phi) >= 1):
        raise ValueError("AR innovations and stable coefficients must align")
    state = np.empty_like(innovations, dtype=float)
    state[0] = innovations[0]
    for index in range(1, len(state)):
        state[index] = phi[index] * state[index - 1] + np.sqrt(1 - phi[index] ** 2) * innovations[index]
    return state


def generate_pair(rate: float, generator: str, noise: float, seed: int,
                  duration: float = 1200, episode: float = 60, ramp: float = 5):
    """Pair each changed trace with identical innovations/observation noise."""
    if generator not in GENERATORS or rate <= 0 or noise < 0 or not 0 < episode < duration:
        raise ValueError("Invalid synthetic trace specification")
    n = int(round(duration * rate))
    times = np.arange(n, dtype=float) / rate
    start, end = (duration - episode) / 2, (duration + episode) / 2
    envelope = episode_envelope(times, start, end, ramp)
    streams = np.random.SeedSequence([seed, int(round(rate * 100))]).spawn(2)
    rng = np.random.default_rng(streams[0])
    observation_noise = noise * np.random.default_rng(streams[1]).standard_normal(n)
    if generator == "ar1":
        innovations = rng.standard_normal(n)
        baseline = ar_process(innovations, np.full(n, 0.5))
        changed = ar_process(innovations, 0.5 + 0.4 * envelope)
        change = {"baseline_phi": 0.5, "episode_phi": 0.9,
                  "innovation_scaling": "sqrt(1-phi^2); unit variance at constant phi"}
    else:
        frequency = 0.2 if generator == "sine_0p2" else 1.0
        baseline = np.sin(2 * np.pi * frequency * times)
        instantaneous_frequency = frequency * (1 + 0.25 * envelope)
        phase = np.zeros(n)
        phase[1:] = 2 * np.pi * np.cumsum(instantaneous_frequency[:-1]) / rate
        changed = np.sin(phase)
        change = {"baseline_hz": frequency, "episode_hz": 1.25 * frequency,
                  "phase": "Integrated frequency; phase continuous, with post-episode phase offset preserved"}
    return baseline + observation_noise, changed + observation_noise, {
        "rate_hz": rate, "generator": generator, "noise_sd": noise, "seed": seed,
        "duration_seconds": duration, "episode_start_seconds": start,
        "episode_end_seconds": end, "ramp_seconds": ramp, "change": change,
        "preprocessing": "Generated directly at feature-input rate; no extra filter/resampling",
    }


def summarize_pair(base: dict, changed: dict, length: float, rate: float, metadata: dict):
    """Keep uncomputable diagnostics missing; never average around undefined features."""
    if not np.array_equal(base["starts"], changed["starts"]):
        raise ValueError("Paired synthetic traces retained different windows")
    centres = base["starts"] / rate + length / 2
    start, end = metadata["episode_start_seconds"], metadata["episode_end_seconds"]
    outside = (centres + length / 2 <= start) | (centres - length / 2 >= end)
    inside = (centres >= start) & (centres < end)
    contained = (centres - length / 2 >= start) & (centres + length / 2 <= end)
    crossing = ~outside & ~contained
    n_features = len(base["feature_names"])
    def variability(values):
        if len(values) < 2:
            return np.full(n_features, np.nan)
        return np.std(values, axis=0, ddof=1)
    effect = (np.mean(changed["values"][inside] - base["values"][inside], axis=0)
              if inside.any() else np.full(n_features, np.nan))
    return {
        "outside_windows": int(outside.sum()), "episode_centre_windows": int(inside.sum()),
        "episode_contained_windows": int(contained.sum()), "crossing_windows": int(crossing.sum()),
        "baseline_outside_sd": variability(base["values"][outside]),
        "changed_outside_sd": variability(changed["values"][outside]),
        "episode_paired_mean_delta": effect,
        "aggregate_delta": changed["aggregates"] - base["aggregates"],
    }


def run_trace(job: tuple) -> dict:
    """Extract a complete paired trace; checkpoint only when every length succeeds."""
    rate, generator, noise, seed, output_dir, signature = job
    output_dir = Path(output_dir)
    stem = trace_stem(rate, generator, noise, seed)
    saved = verified_trace(output_dir, stem, signature)
    if saved:
        return saved
    # A missing final metadata checkpoint means this trace was interrupted.
    # Remove only its own generated artifacts; complete traces are never replaced.
    for suffix in (".npz", ".csv", ".npz.partial", ".csv.partial", ".json.partial"):
        (output_dir / f"{stem}{suffix}").unlink(missing_ok=True)
    from scripts.window_duration_pilot import extract_grid

    baseline, changed, metadata = generate_pair(rate, generator, noise, seed)
    arrays, rows, counts = {}, [], {}
    for length in LENGTHS:
        base = extract_grid(baseline, rate, length, origin=0)
        altered = extract_grid(changed, rate, length, origin=0)
        diagnostic = summarize_pair(base, altered, length, rate, metadata)
        counts[str(length)] = {key: value for key, value in diagnostic.items() if isinstance(value, int)}
        arrays[f"L{length}_starts"] = base["starts"]
        arrays[f"L{length}_candidate_starts"] = base["candidate_starts"]
        for label, result in [("baseline", base), ("changed", altered)]:
            arrays[f"L{length}_{label}_values"] = result["values"]
            arrays[f"L{length}_{label}_aggregates"] = result["aggregates"]
        for index, feature in enumerate(base["feature_names"]):
            record = {"rate_hz": rate, "generator": generator, "noise_sd": noise, "seed": seed,
                      "length_seconds": length, "feature": feature, **counts[str(length)],
                      "baseline_nonfinite_windows": int((~np.isfinite(base["values"][:, index])).sum()),
                      "changed_nonfinite_windows": int((~np.isfinite(altered["values"][:, index])).sum()),
                      "baseline_outside_sd": diagnostic["baseline_outside_sd"][index],
                      "changed_outside_sd": diagnostic["changed_outside_sd"][index],
                      "episode_paired_mean_delta": diagnostic["episode_paired_mean_delta"][index]}
            for reducer, delta in zip(REDUCERS, diagnostic["aggregate_delta"]):
                record[f"aggregate_{reducer}_delta"] = delta[index]
            rows.append(record)
    arrays["feature_names"] = np.asarray(base["feature_names"], dtype=str)
    partial = output_dir / f"{stem}.npz.partial"
    with partial.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    partial.rename(output_dir / f"{stem}.npz")
    partial_csv = output_dir / f"{stem}.csv.partial"
    with partial_csv.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    partial_csv.replace(output_dir / f"{stem}.csv")
    metadata.update(window_counts=counts, lengths_seconds=LENGTHS, origin_samples=0,
                    nominal_overlap=0.5, feature_rows=len(rows),
                    protocol_sha256=signature,
                    artifact_sha256={f"{stem}{suffix}": sha256(output_dir / f"{stem}{suffix}")
                                     for suffix in (".npz", ".csv")},
                    caution="Descriptive feature challenge; absence of contained/centred windows stays missing. Stability is not predictive validity.")
    write_json(output_dir / f"{stem}.json", metadata)
    return {"trace": stem, "feature_rows": len(rows), "window_counts": counts, "reused": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed paired traces")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    protocol = {"protocol_version": 1, "paired_conditions": 2, "duration_seconds": 1200,
                "episode_seconds": 60, "ramp_seconds": 5,
                "script_sha256": sha256(Path(__file__)),
                "feature_script_sha256": sha256(Path(__file__).with_name("window_duration_pilot.py")),
                "numpy_version": np.__version__,
                "pycatch22_version": importlib.metadata.version("pycatch22"),
                "lengths_seconds": list(LENGTHS), "rates_hz": list(RATES),
                "generators": list(GENERATORS), "noise_levels": list(NOISE_LEVELS),
                "seeds": list(range(10)), "no_outcomes_or_models": True}
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    existed = args.output_dir.exists()
    if existed and not args.resume:
        raise ValueError("Output directory already exists; use --resume to verify its checkpoints")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    with (args.output_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.output_dir / "manifest.json"
        if existed:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("protocol") != protocol or manifest.get("protocol_sha256") != signature:
                raise ValueError("Synthetic source code, feature code or protocol changed; use a fresh output directory")
            if manifest.get("status") not in {"running", "failed", "complete", "drained"}:
                raise ValueError("Unrecognized synthetic checkpoint status")
        else:
            manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                        "protocol": protocol, "protocol_sha256": signature}
        jobs = [(rate, generator, noise, seed, str(args.output_dir), signature)
                for rate in RATES for generator in GENERATORS for noise in NOISE_LEVELS for seed in range(10)]
        if manifest.get("status") == "complete":
            for rate, generator, noise, seed, _, _ in jobs:
                if verified_trace(args.output_dir, trace_stem(rate, generator, noise, seed), signature) is None:
                    raise ValueError("Completed synthetic run lost a trace checkpoint")
            print(f"Verified complete synthetic run: {len(jobs)} paired traces", flush=True)
            return
        manifest.update(status="running", traces=len(jobs), workers=args.workers)
        write_json(manifest_path, manifest)
        try:
            def collect(results):
                for index, result in enumerate(results, start=1):
                    action = "verified" if result["reused"] else "complete"
                    print(f"Synthetic trace {index}/{len(jobs)} {action}: {result['trace']}", flush=True)
            if args.adaptive:
                collect(adaptive_map(run_trace, jobs, max_workers=args.workers, volume=args.output_dir,
                    progress_path=args.output_dir / "resources.json",
                    stop_admission_path=args.output_dir.parent / "control/drain-pilot"))
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    collect(pool.map(run_trace, jobs))
            manifest.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat())
        except PilotDrained:
            manifest.update(status="drained", finished_at_utc=datetime.now(timezone.utc).isoformat())
            raise SystemExit(75)
        except Exception:
            manifest.update(status="failed", finished_at_utc=datetime.now(timezone.utc).isoformat())
            raise
        finally:
            write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
