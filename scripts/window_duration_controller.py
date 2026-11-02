"""Resume the frozen feature-only duration pilot, summaries and synthetic audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts.window_duration_pilot import sha, validate_channel_checkpoint, write_json
from scripts.window_duration_summary import summarize


SOURCE = Path(__file__).resolve().parents[1]


def verify_freeze(root, manifest, manifest_hash):
    if SOURCE != (root / "code").resolve():
        raise ValueError("Run this controller from the pilot's frozen code snapshot")
    if sha(root / "control/manifest.json") != manifest_hash:
        raise ValueError("Pilot manifest changed during execution")
    own_path = "scripts/window_duration_controller.py"
    if own_path not in manifest["frozen_source_sha256"]:
        raise ValueError("Controller is missing from the frozen source inventory")
    if "scripts/window_duration_failures.py" not in manifest["frozen_source_sha256"]:
        raise ValueError("Fallback detector is missing from the frozen source inventory")
    for name, digest in manifest["frozen_source_sha256"].items():
        if sha(SOURCE / name) != digest:
            raise ValueError(f"Frozen pilot code changed: {name}")
    for name, digest in manifest["input_sha256"].items():
        if sha(name) != digest:
            raise ValueError(f"Frozen pilot input changed: {name}")
    for name, expected in manifest["packages"].items():
        if version(name) != expected:
            raise ValueError(f"Frozen pilot package changed: {name}")


def status(root, stage, state="running", **extra):
    write_json(root / "control/controller-status.json", {
        "status": state, "stage": stage, "pid": os.getpid(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), **extra})
    print(f"Controller {state}: {stage}", flush=True)


def run_module(module, arguments):
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    subprocess.run([sys.executable, "-u", "-m", module, *arguments],
                   cwd=SOURCE, env=environment, check=True)


def verify_pilot_assembly(root, manifest=None):
    """Read-only verification against channel checkpoints; return metadata/arrays.

    Each assembled array and each channel array is decompressed only once.
    This helper can audit completed historical pilots without running their
    controller or modifying their frozen code, status or outputs.
    """
    root = Path(root)
    if manifest is None:
        manifest = json.loads((root / "control/manifest.json").read_text())
    fields = ("aggregates", "audited_aggregates", "matched", "phases")
    with np.load(root / "pilot.npz", allow_pickle=False) as bundle:
        for field in ("lengths", "channels", "feature_names"):
            if bundle[field].tolist() != manifest[field]:
                raise ValueError(f"Pilot output metadata differs from frozen manifest: {field}")
        for field, sample_field in [("patient_ids", "patient_id"), ("case_ids", "case_id")]:
            if bundle[field].tolist() != [sample[sample_field] for sample in manifest["samples"]]:
                raise ValueError(f"Pilot output identities differ from frozen sample: {field}")
        arrays = {field: bundle[field] for field in fields}
    shape = (manifest["n_patients"], len(manifest["channels"]), len(manifest["lengths"]), 4, 4, 24)
    for field, array in arrays.items():
        expected = (3, *shape) if field == "phases" else shape
        if array.shape != expected:
            raise ValueError(f"Pilot aggregate axes differ from frozen sample: {field}")
    checkpoints = []
    for sample in manifest["samples"]:
        index = sample["index"]
        for ci, channel in enumerate(manifest["channels"]):
            path = root / "channels" / f"{index:03d}__{ci}.npz"
            checkpoints.append(validate_channel_checkpoint(path, index, channel))
            with np.load(path, allow_pickle=False) as checkpoint:
                for field in fields:
                    assembled = arrays[field][:, index, ci] if field == "phases" else arrays[field][index, ci]
                    if not np.array_equal(assembled, checkpoint[field], equal_nan=True):
                        raise ValueError(f"Pilot assembly differs from verified channel checkpoint: {path.name}/{field}")
    return checkpoints, arrays


def write_summary(root, name, arrays, metadata, input_hash, bootstrap):
    directory = root / "summary" / name
    marker = directory / "complete.json"
    identity = {"pilot_sha256": input_hash, "bootstrap": bootstrap, "seed": 0,
                "summary_script_sha256": sha(SOURCE / "scripts/window_duration_summary.py")}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity or any(sha(directory / path) != digest
                                               for path, digest in saved["outputs"].items()):
            raise ValueError(f"Summary checkpoint changed: {name}")
        return json.loads((directory / "summary.json").read_text())
    result, metrics, availability = summarize(
        arrays, metadata["lengths"], channels=metadata["channels"],
        feature_names=metadata["feature_names"], bootstrap=bootstrap, seed=0)
    result.update(audit_variant=name, input={"path": str(root / "pilot.npz"), "sha256": input_hash})
    directory.mkdir(parents=True, exist_ok=True)
    for filename, frame in [("metrics.csv", metrics), ("availability.csv", availability)]:
        partial = directory / (filename + ".partial")
        frame.to_csv(partial, index=False)
        partial.replace(directory / filename)
    write_json(directory / "summary.json", result)
    write_json(marker, {"identity": identity, "outputs": {
        name: sha(directory / name) for name in ["metrics.csv", "availability.csv", "summary.json"]}})
    return result


def make_report(root, manifest, primary, checkpoints, synthetic_complete=False):
    candidate = primary["technical_candidate_seconds"]
    bootstrap = primary["bootstrap"]
    frequency = bootstrap["point_candidate_selection_frequency"]
    failures = sum(bool(row.get("error")) for row in checkpoints)
    verified = sum(row.get("baseline_check") == "verified_against_frozen_10s_origin0" for row in checkpoints)
    lines = ["# Window-duration feature audit", "",
             "Status: complete." if synthetic_complete else "Status: primary summaries complete; synthetic checks still pending. This is an interim report.", "",
             f"The audit sampled {manifest['n_patients']} distinct patients, one operation each, from outer fold "
             f"{manifest['outer_training_fold']}'s main-shared training partition. No held-out patients, outcomes or models informed duration screening.", "",
             f"Channel checkpoints: {len(checkpoints)}; extraction failures: {failures}. "
             f"Frozen 10-second/origin-zero descriptor comparisons passed for {verified} channels. "
             "Channels without a comparison remain documented in the checkpoint metadata.", ""]
    lines += ["Qualification uses the audited aggregates: verified finite algorithmic failure fallbacks are marked missing for availability and origin-robustness checks. "
              "Raw canonical Catch24 values remain unchanged in pilot.npz and have a separate finite-only summary. "
              "The masks follow implementation failure branches in pinned pycatch22 0.4.5; they introduce no clinical cutoff or replacement feature values.", "",
              "Audit-method provenance: [failure detector](code/scripts/window_duration_failures.py), "
              "with its source hash and the pinned package versions recorded in [the frozen manifest](control/manifest.json).", ""]
    lines += ["| Channel | Operations with any verified fallback | Affected length/origin/descriptor cells |",
              "|---|---:|---:|"]
    for channel in primary["channels"]:
        affected = [int(row.get("fallback_affected", 0)) for row in checkpoints if row.get("channel") == channel]
        lines.append(f"| {channel} | {sum(count > 0 for count in affected)} | {sum(affected)} |")
    lines += ["", "Affected cells are counted across operations; they are not independent windows or clinical events.", ""]
    if candidate is None:
        lines.append("No duration met the technical adjacent-doubling and plateau rules.")
    else:
        lines.append(f"The point technical candidate is {candidate} seconds; it was selected in "
                     f"{frequency:.1%} of {bootstrap['replicates']:,} paired patient bootstrap draws. "
                     f"Decision: {primary['decision']}.")
    lines += ["", "A counts descriptors available for at least 95% of sampled operations (gate: 22/24). "
              "B lists origin-robust descriptor counts in mean/SD/min/max order (gate: 20/24 each). "
              "Both gates must pass in every channel. These are proposed engineering tolerances.", "",
              "| Seconds | " + " | ".join(primary["channels"]) + " |",
              "|---|" + "---|" * len(primary["channels"])]
    for index, length in enumerate(primary["lengths"]):
        cells = []
        for channel in range(len(primary["channels"])):
            a = primary["availability_pass_counts"][channel][index]
            b = "/".join(str(x) for x in primary["origin_pass_counts"][channel][index])
            cells.append(f"A {a}; B {b}")
        lines.append(f"| {length} | " + " | ".join(cells) + " |")
    lines += ["", "Bootstrap selections: " + "; ".join(
        f"{key}: {count}/{bootstrap['replicates']}" for key, count in bootstrap["selection_counts"].items() if count), "",
        "The raw finite-only, audited matched-count and audited first/middle/final-third summaries are secondary diagnostics; they do not change the primary selection.", ""]
    if synthetic_complete:
        synthetic = json.loads((root / "synthetic/manifest.json").read_text())
        lines.append(f"Synthetic audit: {synthetic['traces']} paired traces completed across both sampling rates, "
                     "three generators and two noise levels. It tests known changes descriptively, without a clinical attenuation or latency gate.")
        frames = [pd.read_csv(path) for path in sorted((root / "synthetic").glob("rate_*.csv"))]
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            counts = combined.groupby("length_seconds").episode_contained_windows.max()
            missing = counts[counts.eq(0)].index.tolist()
            if missing:
                lines.append("No window fits entirely inside the 60-second episode at lengths "
                             + ", ".join(str(int(x)) for x in missing) + " seconds; those contrasts cannot establish within-episode feature fidelity.")
    else:
        lines.append("Synthetic feature extraction is pending; no physiological duration recommendation is made.")
    lines += ["", "This assesses sensitivity to grid origin, conditional on these recordings. Stable summaries do not demonstrate unbiased feature estimates, "
              "physiological optimality or improved AKI/ICU/mortality prediction. Frozen study code, inputs and labels were preserved. "
              "This one-training-partition screen cannot select a global duration for confirmatory evaluation across all existing folds.", "",
              "Detailed results: [primary](summary/primary/summary.json), [descriptor metrics](summary/primary/metrics.csv), "
              "[availability](summary/primary/availability.csv), [raw finite-only](summary/finite_only/summary.json), "
              "[audited matched counts](summary/matched/summary.json), "
              "[first third](summary/first/summary.json), [middle third](summary/middle/summary.json), "
              "[final third](summary/final/summary.json), and [synthetic provenance](synthetic/manifest.json).", ""]
    partial = root / "REPORT.md.partial"
    partial.write_text("\n".join(lines))
    partial.replace(root / "REPORT.md")


def execute(root, workers, synthetic_workers, adaptive=False):
    manifest_path = root / "control/manifest.json"
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    verify_freeze(root, manifest, manifest_hash)
    stage = "features"
    try:
        status(root, stage)
        feature_status = json.loads((root / "control/status.json").read_text())
        if feature_status.get("status") != "features_complete" or not (root / "pilot.npz").is_file():
            run_module("scripts.window_duration_pilot", ["run", "--root", str(root), "--workers", str(workers)]
                       + (["--adaptive"] if adaptive else []))
        verify_freeze(root, manifest, manifest_hash)
        input_hash = sha(root / "pilot.npz")
        checkpoints, assembled = verify_pilot_assembly(root, manifest)
        variants = [("primary", assembled["audited_aggregates"], 2000),
                    ("finite_only", assembled["aggregates"], 0), ("matched", assembled["matched"], 0)]
        variants += [(name, assembled["phases"][index], 0)
                     for index, name in enumerate(["first", "middle", "final"])]
        for name, arrays, draws in variants:
            stage = "summary_" + name
            status(root, stage)
            result = write_summary(root, name, arrays, manifest, input_hash, draws)
            if name == "primary":
                primary = result
            verify_freeze(root, manifest, manifest_hash)
        if sha(root / "pilot.npz") != input_hash:
            raise ValueError("Pilot aggregate input changed during summaries")
        make_report(root, manifest, primary, checkpoints)
        stage = "synthetic"
        status(root, stage)
        arguments = ["--output-dir", str(root / "synthetic"), "--workers", str(synthetic_workers)]
        if adaptive:
            arguments.append("--adaptive")
        if (root / "synthetic").exists():
            arguments.append("--resume")
        run_module("scripts.window_duration_synthetic", arguments)
        verify_freeze(root, manifest, manifest_hash)
        synthetic = json.loads((root / "synthetic/manifest.json").read_text())
        if synthetic.get("status") != "complete":
            raise ValueError("Synthetic audit did not complete")
        make_report(root, manifest, primary, checkpoints, synthetic_complete=True)
        verify_freeze(root, manifest, manifest_hash)
        status(root, "complete", state="complete", pilot_sha256=input_hash,
               report_sha256=sha(root / "REPORT.md"), no_models_fitted=True)
    except BaseException as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.returncode == 75:
            status(root, stage, state="drained")
            raise SystemExit(75)
        status(root, stage, state="failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--synthetic-workers", type=int, default=2)
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.synthetic_workers) < 1:
        parser.error("Workers must be positive")
    root = args.root.resolve()
    with (root / "control/.controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A controller is already active for this pilot") from exc
        execute(root, args.workers, args.synthetic_workers, args.adaptive)


if __name__ == "__main__":
    main()
