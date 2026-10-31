import os
import logging
import hashlib
import json
from importlib.metadata import version
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import pycatch22
from tqdm import tqdm
import time

from data_preparation.artifact_metadata import (
    STEP_01_COHORT_ARTIFACT,
    ArtifactCompatibilityError,
    read_versioned_csv,
    metadata_path_for,
)
from data_preparation.inputs import (
    COHORT_FILE, 
    CATCH_22_FILE,
    CATCH_22_ERROR_FILE,
    MANDATORY_WAVEFORMS, 
    WAVEFORM_SUBSTITUTIONS, 
    WIN_SEC,
    SLIDE_SEC,
    GENERATE_FULL_FEATURES,
    GENERATE_WINDOWED_FEATURES,
    FULL_FEATURE_TARGET_SR,
    CATCH_22_WINDOWED_FILE,
)
from data_preparation.waveform_processing import (
    WAVEFORM_SPECS, RESAMPLING_POLICY, process_signal, harmonize_sr, load_and_validate_case,
)
from runtime_resources import available_cpu_count

COHORT_REQUIRED_COLUMNS = ["caseid", "opstart", "opend"]
EXTRACTION_PROTOCOL_VERSION = 3


def extraction_worker_count(task_count: int) -> int:
    """Bound extraction processes by affinity, cgroup quota, and pending work."""
    available = available_cpu_count()
    requested = int(os.getenv("AKI_EXTRACTION_WORKERS", str(available)))
    if requested < 1 or task_count < 1:
        raise ValueError("AKI_EXTRACTION_WORKERS and task_count must be positive.")
    return min(available, requested, task_count)


def build_extraction_fingerprint(cohort_df: pd.DataFrame, mode: str) -> dict:
    """Identify signal inputs/settings before permitting checkpoint reuse."""
    roster = cohort_df[COHORT_REQUIRED_COLUMNS].sort_values("caseid").to_csv(index=False)
    code_files = [Path(__file__), Path(__file__).with_name("waveform_processing.py")]
    return {
        "extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION,
        "mode": mode,
        "cohort_roster_sha256": hashlib.sha256(roster.encode()).hexdigest(),
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files},
        "pycatch22_version": version("pycatch22"),
        "vitaldb_version": version("vitaldb"),
        "scipy_version": version("scipy"),
        "resampling_policy": dict(RESAMPLING_POLICY),
        "waveforms": list(MANDATORY_WAVEFORMS),
        "substitutions": WAVEFORM_SUBSTITUTIONS,
        "signal_specs": WAVEFORM_SPECS,
        "full_target_sr": FULL_FEATURE_TARGET_SR,
        "window_seconds": WIN_SEC if mode == "windowed" else None,
        "slide_seconds": SLIDE_SEC if mode == "windowed" else None,
        "max_missing_fraction": 0.05,
        "invalid_sample_policy": "nonfinite_as_missing_interpolate_up_to_5_pct",
        "signal_validity_policy": "reject_empty_or_nonfinite_after_filtering_and_resampling",
        "missing_feature_policy": "preserve_nan",
        "invalid_descriptor_policy": "nonfinite_to_nan_float64_nan_propagating_aggregates_ddof_1",
    }


def load_processed_pairs(output_path: Path, error_path: Path, fingerprint: dict) -> set:
    """Fail closed on unidentified historical extraction outputs or duplicates."""
    output_path, error_path = Path(output_path), Path(error_path)
    if not output_path.exists() and not error_path.exists():
        return set()
    meta_path = metadata_path_for(output_path)
    if not meta_path.exists():
        raise ArtifactCompatibilityError(
            f"Waveform output {output_path} has no extraction fingerprint; use a fresh output root."
        )
    payload = json.loads(meta_path.read_text())
    if payload.get("extraction_fingerprint") != fingerprint:
        raise ArtifactCompatibilityError(
            f"Waveform extraction fingerprint mismatch for {output_path}; use a fresh output root."
        )
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path)
    if existing[["caseid", "waveform"]].isna().any().any():
        raise ArtifactCompatibilityError(f"Missing case/channel identity in {output_path}.")
    if existing.duplicated(["caseid", "waveform"]).any():
        raise ArtifactCompatibilityError(f"Duplicate case/channel pairs in {output_path}.")
    return set(zip(existing["caseid"], existing["waveform"]))


def _save_mode_checkpoint(conf: dict, fingerprint: dict) -> None:
    """Atomically replace complete checkpoints, leaving feature NaNs intact.

    Snapshot replacement costs more I/O than append, but a killed writer can
    never leave a truncated feature record that looks like a valid NaN row.
    """
    output_path, error_path = Path(conf["file"]), Path(conf["error_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for key, path in (("features", output_path), ("errors", error_path)):
        records = conf[key]
        if records:
            frame = pd.DataFrame(records)
            columns = ["caseid", "waveform"] + [c for c in frame if c not in ("caseid", "waveform")]
            frame = frame[columns]
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = pd.read_csv(path, float_precision="round_trip")
                frame = pd.concat([existing, frame], ignore_index=True)
            if key == "features" and frame.duplicated(["caseid", "waveform"]).any():
                raise ArtifactCompatibilityError(f"Duplicate case/channel pairs in checkpoint {path}.")
            if key == "features":
                feature_columns = [column for column in frame if column not in ("caseid", "waveform")]
                # Serialize feature values consistently across checkpoint batches.
                frame[feature_columns] = frame[feature_columns].astype(np.float64)
            temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(temporary, index=False, float_format="%.17g")
            temporary.replace(path)
            conf[key] = []
    payload = {"extraction_protocol_version": EXTRACTION_PROTOCOL_VERSION,
               "extraction_fingerprint": fingerprint,
               "error_file": str(error_path),
               "requested_caseids": conf.get("requested_caseids", []),
               "successful_pairs": len(conf["processed_pairs"])}
    meta_path = metadata_path_for(output_path)
    temporary = meta_path.with_suffix(meta_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(meta_path)

# Validation
if GENERATE_WINDOWED_FEATURES:
    assert WIN_SEC is not None and SLIDE_SEC is not None, "WIN_SEC and SLIDE_SEC must be set for windowed extraction."

def _process_case(case: Tuple[int, float, float, str]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Processes a single case/waveform pair.
    Returns:
        - results: Dict with keys 'full' and/or 'windowed' containing feature dicts.
        - timings: Dict of timing metrics.
    """
    caseid, opstart, opend, waveform_key = case
    timings = {}
    results = {}
    
    start = time.perf_counter()

    try:
        # 1. Load waveform with substitutions logic
        wave, spec_key_loaded, error_msg = load_and_validate_case(caseid, waveform_key, WAVEFORM_SUBSTITUTIONS)
        
        if error_msg:
             err = {'caseid': caseid, 'waveform': waveform_key, 'error': error_msg}
             return {'error': err}, timings
        
        spec = WAVEFORM_SPECS[spec_key_loaded]
        end = time.perf_counter()
        timings['load_case'] = end - start
        start = end

        # 5. Slicing at NATIVE sample rate
        native_sr = spec['native_sr']
        seg = wave[int(opstart * native_sr):int(opend * native_sr)]
        seg = seg.squeeze()
        
        # 6. Treat all nonfinite samples as missing under the same 5% rule.
        if seg.size == 0:
            err = {'caseid': caseid, 'waveform': waveform_key, 'error': 'invalid_signal_empty'}
            return {'error': err}, timings
        invalid_mask = ~np.isfinite(seg)
        invalid_pct = np.mean(invalid_mask)
        if invalid_pct > 0.05:
            err = {'caseid': caseid, 'waveform': waveform_key, 'error': 'invalid_signal_gt_5_pct_nonfinite'}
            return {'error': err}, timings

        if invalid_pct > 0:
            x = np.arange(seg.size)
            seg = np.interp(x=x, xp=x[~invalid_mask], fp=seg[~invalid_mask])
        
        # 7. & 8. Filtering and Resampling
        seg, error_msg = process_signal(seg, spec, caseid, waveform_key)
        if error_msg:
             err = {'caseid': caseid, 'waveform': waveform_key, 'error': error_msg}
             return {'error': err}, timings

        target_sr = spec['target_sr']

        # === FULL FEATURE EXTRACTION ===
        if GENERATE_FULL_FEATURES:
            start_full = time.perf_counter()
            if np.isnan(seg).all() or np.nanstd(seg) < 1e-6: 
                # Log error but don't fail the whole case if windowed might work (unlikely but possible)
                results['full'] = {'caseid': caseid, 'waveform': waveform_key, 'error': 'invalid_signal_flatline_or_all_nan'}
            else:
                # Keep the 10 Hz full-series limit with anti-alias resampling.
                seg_full = harmonize_sr(seg, target_sr, FULL_FEATURE_TARGET_SR)
                if seg_full.size == 0 or not np.isfinite(seg_full).all():
                    results['full'] = {'caseid': caseid, 'waveform': waveform_key,
                                       'error': 'invalid_full_resampled_signal_empty_or_nonfinite'}
                else:
                    all_feature_results = pycatch22.catch22_all(seg_full, catch24=True)
                    out = {n: v if np.isfinite(v) else np.nan
                           for n, v in zip(all_feature_results['names'], all_feature_results['values'])}
                    out['caseid'] = caseid
                    out['waveform'] = waveform_key
                    results['full'] = out
                
            timings['process_full'] = time.perf_counter() - start_full

        # === WINDOWED FEATURE EXTRACTION ===
        if GENERATE_WINDOWED_FEATURES:
            start_win = time.perf_counter()
            win_samp = int(WIN_SEC * target_sr)
            slide_samp = int(SLIDE_SEC * target_sr)
            
            if seg.size < win_samp:
                results['windowed'] = {'caseid': caseid, 'waveform': waveform_key, 'error': 'segment_too_short_for_window'}
            else:
                feats_list = []
                
                # Pre-calculate window indices
                window_starts = range(0, seg.size - win_samp + 1, slide_samp)
                
                for i in window_starts:
                    win = seg[i:i + win_samp]
                    is_valid = (not np.isnan(win).all()) and (np.nanstd(win) >= 1e-6)
                    
                    if is_valid:
                        all_feature_results = pycatch22.catch22_all(win, catch24=True)
                        feats_list.append(all_feature_results)

                if not feats_list:
                    results['windowed'] = {'caseid': caseid, 'waveform': waveform_key, 'error': 'no_valid_windows_found'}
                else:
                    feature_names = feats_list[0]['names']
                    feature_values = [res['values'] for res in feats_list]
                    m = np.asarray(feature_values, dtype=np.float64)
                    m[~np.isfinite(m)] = np.nan
                    # Keep undefined-window propagation and sample SD unchanged.
                    # Float64 avoids overflow from finite float32-sized descriptors.
                    with np.errstate(over='ignore', invalid='ignore'):
                        out = {
                            **{f'{n}_mean': v for n, v in zip(feature_names, m.mean(0))},
                            **{f'{n}_std':  v for n, v in zip(feature_names, m.std(0, ddof=1))},
                            **{f'{n}_min':  v for n, v in zip(feature_names, m.min(0))},
                            **{f'{n}_max':  v for n, v in zip(feature_names, m.max(0))}
                        }
                    out = {n: v if np.isfinite(v) else np.nan for n, v in out.items()}
                    out['caseid'] = caseid
                    out['waveform'] = waveform_key
                    results['windowed'] = out

            timings['process_windowed'] = time.perf_counter() - start_win

        return results, timings

    except Exception as e:
        logging.error(f"Error processing case {caseid} waveform {waveform_key}: {e}")
        err = {'caseid': caseid, 'waveform': waveform_key, 'error': str(e)}
        return {'error': err}, timings

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting catch22 feature extraction...")

    # Define outputs based on configuration
    outputs_config = {}
    if GENERATE_FULL_FEATURES:
        outputs_config['full'] = {
            'file': CATCH_22_FILE,
            'error_file': CATCH_22_ERROR_FILE,
            'features': [],
            'errors': []
        }
    if GENERATE_WINDOWED_FEATURES:
        def _add_suffix(filepath, suffix):
            name, ext = os.path.splitext(filepath)
            return f"{name}{suffix}{ext}"
        
        outputs_config['windowed'] = {
            'file': CATCH_22_WINDOWED_FILE,
            'error_file': _add_suffix(CATCH_22_ERROR_FILE, '_windowed'),
            'features': [],
            'errors': []
        }

    for mode, conf in outputs_config.items():
        logging.info(f"Mode '{mode}': Output={conf['file']}, Error={conf['error_file']}")

    # Load Cohort
    try:
        cohort_df, _ = read_versioned_csv(
            COHORT_FILE,
            artifact_role=STEP_01_COHORT_ARTIFACT,
            required_columns=COHORT_REQUIRED_COLUMNS,
        )
    except (FileNotFoundError, ArtifactCompatibilityError) as exc:
        raise SystemExit(str(exc)) from exc
    logging.info(f"Loaded {len(cohort_df)} cases from {COHORT_FILE}")

    # Checkpoint logic: We need to process a pair if it is missing from ANY enabled output
    processed_pairs_by_mode = {}
    for mode, conf in outputs_config.items():
        conf["fingerprint"] = build_extraction_fingerprint(cohort_df, mode)
        conf["requested_caseids"] = sorted(cohort_df["caseid"].tolist())
        processed_pairs = load_processed_pairs(conf["file"], conf["error_file"], conf["fingerprint"])
        conf["processed_pairs"] = processed_pairs
        _save_mode_checkpoint(conf, conf["fingerprint"])
        processed_pairs_by_mode[mode] = processed_pairs
        logging.info(f"Mode '{mode}': Found {len(processed_pairs)} previously processed pairs.")

    # Expand cohort
    wf_df = pd.DataFrame({'waveform': MANDATORY_WAVEFORMS})
    cohort_df['__k'] = 1
    wf_df['__k'] = 1
    cohort_expanded = (cohort_df.merge(wf_df, on='__k', how='left').drop(columns='__k'))
    del cohort_df

    # Determine worklist
    # Skip a pair only when every enabled mode has a saved successful result.
    # Recomputed modes are discarded by the per-mode writer below.
    if not processed_pairs_by_mode:
        universal_processed = set()
    else:
        universal_processed = set.intersection(*processed_pairs_by_mode.values())
    
    if universal_processed:
        processed_pairs_df = pd.DataFrame(list(universal_processed), columns=['caseid', 'waveform'])
        cohort_to_process = (
            cohort_expanded.merge(processed_pairs_df, on=['caseid', 'waveform'], how='left', indicator=True)
                           .loc[lambda df: df['_merge'] == 'left_only']
                           .drop(columns=['_merge'])
        )
    else:
        cohort_to_process = cohort_expanded
    
    del cohort_expanded

    if cohort_to_process.empty:
        logging.info("All pairs processed in all enabled modes. Nothing to do.")
        return

    cases_to_process = list(
        cohort_to_process[['caseid', 'opstart', 'opend', 'waveform']]
        .itertuples(index=False, name=None)
    )
    
    logging.info(f"{len(cohort_to_process)} pairs to process.")

    num_processes = extraction_worker_count(len(cases_to_process))
    checkpoint_pairs = int(os.getenv("AKI_EXTRACTION_CHECKPOINT_PAIRS", "32"))
    if checkpoint_pairs < 1:
        raise ValueError("AKI_EXTRACTION_CHECKPOINT_PAIRS must be positive.")
    
    logging.info(f"Using {num_processes} worker processes.")
    
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(cases_to_process), desc="Extracting Features") as pbar:
            for completed, (result_dict, _timing) in enumerate(pool.imap_unordered(_process_case, cases_to_process), 1):
                # Handle global error (loading failure)
                if 'error' in result_dict:
                    # This error applies to ALL modes
                    for mode in outputs_config:
                        error = result_dict['error']
                        if (error['caseid'], error['waveform']) not in outputs_config[mode]['processed_pairs']:
                            outputs_config[mode]['errors'].append(error)
                else:
                    # Distribute results to respective modes
                    for mode, res in result_dict.items():
                        pair = (res['caseid'], res['waveform'])
                        if pair in outputs_config[mode]['processed_pairs']:
                            continue
                        if 'error' in res:
                            outputs_config[mode]['errors'].append(res)
                        else:
                            outputs_config[mode]['features'].append(res)
                            outputs_config[mode]['processed_pairs'].add(pair)
                if completed % checkpoint_pairs == 0:
                    for conf in outputs_config.values():
                        _save_mode_checkpoint(conf, conf["fingerprint"])
                pbar.update()

    for conf in outputs_config.values():
        _save_mode_checkpoint(conf, conf["fingerprint"])

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
