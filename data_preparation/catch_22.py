import os
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import vitaldb
import pycatch22
from tqdm import tqdm
from data_preparation.inputs import (
    COHORT_FILE, 
    CATCH_22_FILE, 
    MANDATORY_WAVEFORMS, 
    WAVEFORM_SUBSTITUTIONS, 
    WIN_SEC,
    SLIDE_SEC,
    TARGET_SR,
    OUTCOME
)

WIN_SAMP = int(WIN_SEC * TARGET_SR) if WIN_SEC else None
SLIDE_SAMP = int(SLIDE_SEC * TARGET_SR) if SLIDE_SEC else None
assert (WIN_SAMP is None and SLIDE_SAMP is None) or (WIN_SAMP is not None and SLIDE_SAMP is not None), "Invalid window/slide settings"

# TODO: finish this, it's so wrong rn lmao
def _process_case(case: Tuple[int, float, float, int]) -> Dict[str, Any]:
    caseid, opstart, opend, outcome = case
    for waveform in MANDATORY_WAVEFORMS:
        wave = vitaldb.load_case(caseid, waveform, 1 / TARGET_SR)
        for sub_waveform in WAVEFORM_SUBSTITUTIONS.get(waveform, []):
            if wave is None or wave.size == 0:
                wave = vitaldb.load_case(caseid, sub_waveform, 1 / TARGET_SR)
            else:
                break
        if wave is None or wave.size == 0:
            return {'caseid': caseid, 'waveform':waveform, 'error': 'empty'}

        seg = wave[int(opstart*TARGET_SR):int(opend*TARGET_SR)]
        if WIN_SAMP is None and SLIDE_SAMP is None:
            if np.isnan(seg).all() or np.nanstd(seg) < 1e-6:
                return {'caseid': caseid, 'waveform':waveform, 'error': 'invalid_signal'}
            all_feature_results = pycatch22.catch22_all(seg, catch24=True)
            out = {n: v for n, v in zip(all_feature_results['names'], all_feature_results['values'])}
            out['caseid'] = caseid
            out['waveform'] = waveform
            out[OUTCOME] = outcome
            return out
        else:
            assert WIN_SAMP is not None and SLIDE_SAMP is not None
            if seg.size < WIN_SAMP:
                return {'caseid': caseid, 'waveform':waveform, 'error': 'too_short'}

            feats_list = []
            for i in range(0, seg.size - WIN_SAMP + 1, SLIDE_SAMP):
                win = seg[i:i+WIN_SAMP]
                if np.isnan(win).all() or np.nanstd(win) < 1e-6:
                    continue
                all_feature_results = pycatch22.catch22_all(win, catch24=False)
                feats_list.append(all_feature_results)

            if not feats_list:
                return {'caseid': caseid, 'waveform':waveform, 'error': 'no_valid_window'}
            
            feature_names = feats_list[0]['names']
            feature_values = [res['values'] for res in feats_list]

            m = np.asarray(feature_values, dtype=np.float32)
            out = {
                **{f'{n}_mean': v for n, v in zip(feature_names, m.mean(0))},
                **{f'{n}_std':  v for n, v in zip(feature_names, m.std(0, ddof=1))}, # Use ddof=1 for sample std
                **{f'{n}_min':  v for n, v in zip(feature_names, m.min(0))},
                **{f'{n}_max':  v for n, v in zip(feature_names, m.max(0))}
            }
            out['caseid'] = caseid
            out['waveform'] = waveform
            out[OUTCOME] = outcome
            return out
