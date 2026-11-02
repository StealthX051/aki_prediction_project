"""Isolated, outcome-blinded duration pilot; never imports model fitting code."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import pandas as pd
import pycatch22

from scripts.window_duration_failures import library_audit_metadata, verified_fallbacks
from scripts.window_duration_resources import adaptive_map, PilotDrained

from data_preparation.waveform_processing import (
    WAVEFORM_SPECS, load_and_validate_case, process_signal,
)

LENGTHS = (10, 20, 40, 80, 160, 320, 640)
CHANNELS = ("SNUADC/PLETH", "SNUADC/ECG_II", "Primus/CO2", "Primus/AWP")
REDUCERS = ("mean", "sd", "min", "max")
SUBSTITUTIONS = {"SNUADC/ECG_II": ["SNUADC/ECG_V5"]}
FEATURE_NAMES = pycatch22.catch22_all(np.sin(np.arange(1000) * .1), catch24=True)["names"]
SOURCE = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def validate_channel_checkpoint(path, sample_index, channel):
    """Bind a completed checkpoint's checksum to its requested operation/channel."""
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("sample_index") != sample_index or metadata.get("channel") != channel:
        raise ValueError(f"Channel checkpoint identity mismatch: {path.name}")
    if sha(path) != metadata.get("output_sha256"):
        raise ValueError(f"Channel checkpoint hash mismatch: {path.name}")
    return metadata


def aggregate(values):
    if len(values) == 0:
        return np.full((4, 24), np.nan)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        result = np.stack([values.mean(0), values.std(0, ddof=1) if len(values) > 1
                           else np.full(24, np.nan), values.min(0), values.max(0)])
    result[~np.isfinite(result)] = np.nan
    return result


def extract_grid(signal, rate, length, origin=0):
    """Pinned Catch24 and production NaN propagation, with audit diagnostics."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("Expected a finite one-dimensional filtered signal")
    win, stride = int(length * rate), int(length * rate / 2)
    if win < 1 or stride < 1 or origin < 0:
        raise ValueError("Invalid window grid")
    candidates = np.arange(origin, len(signal) - win + 1, stride, dtype=np.int64)
    values, starts, fallbacks = [], [], []
    for start in candidates:
        window = signal[start:start + win]
        if np.std(window) < 1e-6:
            continue
        result = pycatch22.catch22_all(window, catch24=True)
        if result["names"] != FEATURE_NAMES:
            raise RuntimeError("Catch24 descriptor identities changed")
        values.append(result["values"])
        fallbacks.append(verified_fallbacks(window, result["values"], FEATURE_NAMES))
        starts.append(start)
    values = np.asarray(values, dtype=np.float64).reshape(-1, 24)
    values[~np.isfinite(values)] = np.nan
    return {"aggregates": aggregate(values), "values": values,
            "fallbacks": np.asarray(fallbacks, dtype=bool).reshape(-1, 24),
            "starts": np.asarray(starts, dtype=np.int64),
            "candidate_starts": candidates, "feature_names": FEATURE_NAMES}


def prepare(root, baseline, clinical, fold, reuse_filtered_from=None):
    if root.exists():
        raise ValueError("Preparation requires a fresh pilot root")
    if version("pycatch22") != "0.4.5":
        raise ValueError("Pilot requires frozen pycatch22 0.4.5")
    predictions = baseline / "results/catch22/experiments/main_shared/nested_cv/models/logistic_regression/any_aki/windowed/preop_only/predictions"
    train_path, test_path = predictions / "train_oof.csv", predictions / "test.csv"
    columns = ["caseid", "subjectid", "outer_fold_id"]
    train, test = pd.read_csv(train_path, usecols=columns), pd.read_csv(test_path, usecols=columns)
    train = train.loc[train.outer_fold_id.eq(fold)]
    test = test.loc[test.outer_fold_id.eq(fold)]
    if set(train.subjectid) & set(test.subjectid) or train.caseid.duplicated().any():
        raise ValueError("Invalid saved outer patient separation")
    rng = np.random.default_rng(0)
    patients = rng.choice(np.sort(train.subjectid.unique()), 120, replace=False)
    cases = [int(rng.choice(np.sort(train.loc[train.subjectid.eq(patient), "caseid"]))) for patient in patients]
    bounds = pd.read_csv(clinical, usecols=["caseid", "subjectid", "opstart", "opend"])
    if not bounds.caseid.is_unique:
        raise ValueError("Duplicate clinical case IDs")
    bounds = bounds.set_index("caseid")
    samples = []
    for index, (patient, case) in enumerate(zip(patients, cases)):
        row = bounds.loc[case]
        if row.subjectid != patient or not np.isfinite([row.opstart, row.opend]).all() or row.opend <= row.opstart:
            raise ValueError("Invalid sampled operation identity/bounds")
        samples.append(dict(index=index, patient_id=int(patient), case_id=case,
                            opstart=float(row.opstart), opend=float(row.opend)))
    reference = baseline / "data/processed/aki_pleth_ecg_co2_awp_inf_windowed.csv"
    saved = pd.read_csv(reference).loc[lambda frame: frame.caseid.isin(cases)]
    if saved.duplicated(["caseid", "waveform"]).any():
        raise ValueError("Duplicate source descriptor rows")
    source_columns = [name + "_" + reducer for reducer in ("mean", "std", "min", "max") for name in FEATURE_NAMES]
    for sample in samples:
        sample["reference"] = {}
        for channel in CHANNELS:
            row = saved.loc[saved.caseid.eq(sample["case_id"]) & saved.waveform.eq(channel)]
            if len(row):
                values = row[source_columns].to_numpy(dtype=float).reshape(4, 24)
                sample["reference"][channel] = [[float(x) if np.isfinite(x) else None for x in line] for line in values]
    reused = None
    if reuse_filtered_from is not None:
        previous_path = reuse_filtered_from / "control/manifest.json"
        previous = json.loads(previous_path.read_text())
        fields = ["index", "patient_id", "case_id", "opstart", "opend"]
        if ([{key: row[key] for key in fields} for row in previous["samples"]]
                != [{key: row[key] for key in fields} for row in samples]
                or previous["frozen_source_sha256"]["data_preparation/waveform_processing.py"]
                != sha(SOURCE / "data_preparation/waveform_processing.py")
                or any(previous["packages"][name] != version(name) for name in previous["packages"])):
            raise ValueError("Filtered-cache source sample, preprocessing or packages differ")
        reused = {"source_root": str(reuse_filtered_from), "source_manifest_sha256": sha(previous_path), "files": {}}
    root.mkdir(parents=True)
    files = ["scripts/window_duration_pilot.py", "scripts/window_duration_summary.py",
             "scripts/window_duration_synthetic.py", "scripts/window_duration_controller.py",
             "scripts/window_duration_failures.py",
             "scripts/window_duration_resources.py", "runtime_resources.py",
             "model_creation/config_scheduler.py",
             "data_preparation/waveform_processing.py",
             "data_preparation/__init__.py", ".agent/workflows/window-duration-pilot.md"]
    code = root / "code"
    source_hashes = {}
    for name in files:
        source = SOURCE / name
        if source.is_file():
            target = code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_hashes[name] = sha(source)
    inputs = {str(path): sha(path) for path in [clinical, train_path, test_path, reference]}
    if reused is not None:
        (root / "filtered").mkdir()
        for metadata_path in (reuse_filtered_from / "filtered").glob("*.json"):
            cached = json.loads(metadata_path.read_text())
            array_path = metadata_path.with_suffix(".npy")
            if not array_path.is_file() or sha(array_path) != cached["sha256"]:
                raise ValueError("Filtered-cache source checksum mismatch")
            shutil.copy2(array_path, root / "filtered" / array_path.name)
            shutil.copy2(metadata_path, root / "filtered" / metadata_path.name)
            reused["files"][array_path.name] = cached["sha256"]
        inputs[str(reuse_filtered_from / "control/manifest.json")] = reused["source_manifest_sha256"]
    manifest = dict(prepared_at_utc=datetime.now(timezone.utc).isoformat(), scope="main_shared",
                    outcome_partition="any_aki", branch="windowed", outer_training_fold=fold,
                    n_patients=120, lengths=list(LENGTHS), channels=list(CHANNELS), reducers=list(REDUCERS),
                    feature_names=FEATURE_NAMES, input_sha256=inputs, frozen_source_sha256=source_hashes,
                    packages={name: version(name) for name in ["numpy", "scipy", "pandas", "pycatch22", "vitaldb"]},
                    samples=samples, study_labels_used=False, test_patients_excluded=True,
                    waveform_loader="original vitaldb.load_case API path; per-channel filtered cache", seed=0,
                    reused_filtered_cache=reused, finite_failure_audit=library_audit_metadata())
    write_json(root / "control/manifest.json", manifest)
    write_json(root / "control/status.json", {"status": "prepared", "completed_channels": 0, "total_channels": 480})
    print("Prepared 120 training patients; held-out test patients excluded", flush=True)


def process_channel(task):
    root, sample, channel = Path(task[0]), task[1], task[2]
    tag = f"{sample['index']:03d}__{CHANNELS.index(channel)}"
    output = root / "channels" / (tag + ".npz")
    metadata_path = output.with_suffix(".json")
    if output.exists() and metadata_path.exists():
        return validate_channel_checkpoint(output, sample["index"], channel)
    start_time = time.monotonic()
    socket.setdefaulttimeout(120)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = root / "filtered" / (tag + ".npy")
    cache_metadata = cache.with_suffix(".json")
    error, source_key = None, channel
    if cache.exists() and cache_metadata.exists():
        cached = json.loads(cache_metadata.read_text())
        if sha(cache) != cached["sha256"]:
            raise ValueError("Filtered checkpoint hash mismatch")
        signal = np.load(cache, allow_pickle=False)
        source_key = cached["source_key"]
    else:
        signal = np.empty(0)
        for attempt in range(3):
            wave, source_key, error = load_and_validate_case(sample["case_id"], channel, SUBSTITUTIONS)
            if not error:
                break
            if attempt < 2:
                time.sleep(2)
        if not error:
            spec = WAVEFORM_SPECS[source_key]
            signal = np.asarray(wave[int(sample["opstart"] * spec["native_sr"]):
                                     int(sample["opend"] * spec["native_sr"])], dtype=np.float64).reshape(-1)
            missing = ~np.isfinite(signal)
            if signal.size == 0:
                error = "invalid_signal_empty"
            elif missing.mean() > .05:
                error = "invalid_signal_gt_5_pct_nonfinite"
            else:
                if missing.any():
                    positions = np.arange(len(signal))
                    signal = np.interp(positions, positions[~missing], signal[~missing])
                signal, error = process_signal(signal, spec, sample["case_id"], channel)
        if not error:
            cache.parent.mkdir(parents=True, exist_ok=True)
            partial = cache.with_suffix(".partial")
            with partial.open("wb") as stream:
                np.save(stream, signal)
            partial.replace(cache)
            write_json(cache_metadata, {"sha256": sha(cache), "source_key": source_key})
    rate = WAVEFORM_SPECS[source_key or channel]["target_sr"]
    arrays = np.full((7, 4, 4, 24), np.nan)
    audited = arrays.copy()
    matched = arrays.copy()
    phases = np.full((3, 7, 4, 4, 24), np.nan)
    diagnostics = []
    affected = 0
    baseline = "not_available_in_original_extraction"
    if not error:
        reference = sample["reference"].get(channel)
        for li, length in enumerate(LENGTHS):
            win, stride = int(length * rate), int(length * rate / 2)
            origins = [0, stride // 4, stride // 2, 3 * stride // 4]
            grids = []
            for oi, origin in enumerate(origins):
                result = extract_grid(signal, rate, length, origin)
                if length == 10 and oi == 0 and reference is not None:
                    expected = np.asarray(reference, dtype=np.float64)
                    if not np.allclose(result["aggregates"], expected, rtol=1e-8, atol=1e-10, equal_nan=True):
                        raise RuntimeError(f"10-second frozen-extraction mismatch for pilot task {tag}")
                    baseline = "verified_against_frozen_10s_origin0"
                    print(f"Pilot task {tag}: frozen 10-second baseline agrees", flush=True)
                arrays[li, oi] = result["aggregates"]
                qualified = result["values"].copy()
                qualified[result["fallbacks"]] = np.nan
                result["qualified_values"] = qualified
                audited[li, oi] = aggregate(qualified)
                affected += int(np.any(result["fallbacks"], axis=0).sum())
                grids.append(result)
                starts = result["starts"]
                for phase in range(3):
                    mask = (starts >= int(len(signal) * phase / 3)) & (starts + win <= int(len(signal) * (phase + 1) / 3))
                    if mask.sum() >= 2:
                        phases[phase, li, oi] = aggregate(qualified[mask])
                diagnostics.append(dict(length=length, origin_index=oi,
                    candidate_windows=len(result["candidate_starts"]), valid_windows=len(starts),
                    undefined_window_descriptors=np.sum(~np.isfinite(result["values"]), axis=0).tolist(),
                    verified_fallback_windows_per_descriptor=result["fallbacks"].sum(axis=0).tolist(),
                    first_candidate_sample=int(result["candidate_starts"][0]) if len(result["candidate_starts"]) else None,
                    last_candidate_end_sample=int(result["candidate_starts"][-1] + win) if len(result["candidate_starts"]) else None))
            count = min(len(grid["candidate_starts"]) for grid in grids)
            if count:
                for oi, grid in enumerate(grids):
                    mask = grid["starts"] <= grid["candidate_starts"][count - 1]
                    matched[li, oi] = aggregate(grid["qualified_values"][mask])
            print(f"Pilot task {tag}: {length}-second grids complete", flush=True)
    partial = output.with_suffix(".partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, aggregates=arrays, audited_aggregates=audited, matched=matched, phases=phases)
    partial.replace(output)
    metadata = dict(sample_index=sample["index"], channel=channel, source_key=source_key,
                    error=error, rate=rate, filtered_samples=len(signal),
                    elapsed_seconds=time.monotonic() - start_time, baseline_check=baseline,
                    fallback_affected=affected,
                    diagnostics=diagnostics, output_sha256=sha(output))
    write_json(metadata_path, metadata)
    return metadata


def run(root, workers, adaptive=False):
    manifest = json.loads((root / "control/manifest.json").read_text())
    for name, digest in manifest["frozen_source_sha256"].items():
        if sha(SOURCE / name) != digest:
            raise ValueError("Pilot code differs from its frozen preparation snapshot")
    for name, digest in manifest["input_sha256"].items():
        if sha(name) != digest:
            raise ValueError("A frozen pilot input changed")
    jobs = [(str(root), sample, channel) for sample in manifest["samples"] for channel in CHANNELS]
    start = time.monotonic()
    completed = []
    try:
        def collect(results):
            for result in results:
                completed.append(result)
                write_json(root / "control/status.json", dict(status="extracting", completed_channels=len(completed),
                    total_channels=len(jobs), failed_channels=sum(bool(row["error"]) for row in completed),
                    elapsed_seconds=time.monotonic() - start))
                print(f"Channels complete {len(completed)}/{len(jobs)}; failed {sum(bool(r['error']) for r in completed)}", flush=True)
        if adaptive:
            collect(adaptive_map(process_channel, jobs, max_workers=workers, volume=root,
                progress_path=root / "control/resources.json",
                stop_admission_path=root / "control/drain-pilot"))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(process_channel, job) for job in jobs]
                collect(future.result() for future in as_completed(futures))
    except PilotDrained:
        write_json(root / "control/status.json", dict(status="drained", completed_channels=len(completed)))
        raise SystemExit(75)
    except BaseException as exc:
        write_json(root / "control/status.json", dict(status="failed", error=str(exc), completed_channels=len(completed)))
        raise
    arrays = np.full((120, 4, 7, 4, 4, 24), np.nan)
    audited = arrays.copy()
    matched = arrays.copy()
    phases = np.full((3, 120, 4, 7, 4, 4, 24), np.nan)
    for sample in manifest["samples"]:
        for ci, channel in enumerate(CHANNELS):
            path = root / "channels" / f"{sample['index']:03d}__{ci}.npz"
            with np.load(path, allow_pickle=False) as data:
                arrays[sample["index"], ci] = data["aggregates"]
                audited[sample["index"], ci] = data["audited_aggregates"]
                matched[sample["index"], ci] = data["matched"]
                phases[:, sample["index"], ci] = data["phases"]
    with (root / "pilot.npz").open("wb") as stream:
        np.savez_compressed(stream, aggregates=arrays, audited_aggregates=audited, matched=matched, phases=phases,
            lengths=np.asarray(LENGTHS), channels=np.asarray(CHANNELS), feature_names=np.asarray(FEATURE_NAMES),
            patient_ids=np.asarray([s["patient_id"] for s in manifest["samples"]]),
            case_ids=np.asarray([s["case_id"] for s in manifest["samples"]]))
    pd.DataFrame([{"sample_index": row["sample_index"], "channel": row["channel"],
                  "error": row["error"], "elapsed_seconds": row["elapsed_seconds"],
                  "baseline_check": row["baseline_check"], **record}
                 for row in completed for record in row["diagnostics"]]).to_json(root / "grid_diagnostics.json", orient="records")
    write_json(root / "control/status.json", dict(status="features_complete", completed_channels=480,
        failed_channels=sum(bool(row["error"]) for row in completed), elapsed_seconds=time.monotonic() - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--clinical", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--reuse-filtered-from", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--adaptive", action="store_true", help="Treat --workers as a spare-CPU admission ceiling")
    args = parser.parse_args()
    if args.action == "prepare":
        if args.baseline is None or args.clinical is None:
            parser.error("Preparation requires --baseline and --clinical")
        prepare(args.root, args.baseline, args.clinical, args.fold, args.reuse_filtered_from)
    else:
        run(args.root, args.workers, args.adaptive)


if __name__ == "__main__":
    main()
