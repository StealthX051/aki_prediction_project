import numpy as np
import scipy.signal
import logging
from typing import Tuple, Optional, Dict, List, Any
import vitaldb

# --- WAVEFORM SPECIFICATIONS ---
# Define filtering and sampling parameters for each waveform type
WAVEFORM_SPECS = {
    'SNUADC/PLETH': {
        'id': 'SNUADC/PLETH',
        'native_sr': 500,  # Hz
        'target_sr': 100,  # Hz
        'filter_type': 'bandpass',
        'filter_band': [0.1, 10], # Hz
        'filter_order': 4,
        'unit': ' %'
    },
    'SNUADC/ECG_II': {
        'id': 'SNUADC/ECG_II',
        'native_sr': 500,
        'target_sr': 100,
        'filter_type': 'bandpass',
        'filter_band': [0.5, 40],
        'filter_order': 4,
        'unit': 'mV'
    },
    'SNUADC/ECG_V5': {  # Substitute for ECG_II
        'id': 'SNUADC/ECG_V5',
        'native_sr': 500,
        'target_sr': 100,
        'filter_type': 'bandpass',
        'filter_band': [0.5, 40],
        'filter_order': 4,
        'unit': 'mV'
    },
    'Primus/CO2': {
        'id': 'Primus/CO2',
        'native_sr': 62.5,
        'target_sr': 62.5, # Keep native
        'filter_type': 'lowpass',
        'filter_band': 8,
        'filter_order': 4,
        'unit': 'mmHg'
    },
    'Primus/AWP': {
        'id': 'Primus/AWP',
        'native_sr': 62.5,
        'target_sr': 62.5, # Keep native
        'filter_type': 'lowpass',
        'filter_band': 12,
        'filter_order': 4,
        'unit': 'cmH2O'
    }
}

# Create mapping from VitalDB ID to Spec Key for easy lookup
ID_TO_SPEC_KEY = {spec['id']: key for key, spec in WAVEFORM_SPECS.items()}

def harmonize_sr(arr: np.ndarray, current_sr: float, target_sr: float) -> np.ndarray:
    """
    Resamples array from current_sr to target_sr using linear interpolation.
    """
    if current_sr == target_sr:
        return arr
    
    mp = len(arr)
    duration = mp / current_sr
    new_len = int(duration * target_sr)
    
    x_old = np.linspace(0, duration, mp)
    x_new = np.linspace(0, duration, new_len)
    
    return np.interp(x_new, x_old, arr)

def process_signal(seg: np.ndarray, spec: Dict, caseid: int, waveform_key: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    Applies filtering and resampling to a raw signal segment.
    """
    # 1. Filtering
    try:
        if spec['filter_type'] == 'bandpass':
            sos = scipy.signal.butter(spec['filter_order'], spec['filter_band'], btype='bandpass', fs=spec['native_sr'], output='sos')
            seg = scipy.signal.sosfiltfilt(sos, seg)
        elif spec['filter_type'] == 'lowpass':
            sos = scipy.signal.butter(spec['filter_order'], spec['filter_band'], btype='low', fs=spec['native_sr'], output='sos')
            seg = scipy.signal.sosfiltfilt(sos, seg)
    except Exception as e:
        return np.array([]), f"filtering_error: {e}"

    # 2. Resampling (if needed)
    if spec['target_sr'] != spec['native_sr']:
        try:
            # We use harmonize_sr (linear interp) for consistency
            seg = harmonize_sr(seg, spec['native_sr'], spec['target_sr'])
        except Exception as e:
            return np.array([]), f"resampling_error: {e}"
            
    return seg, None

def load_and_validate_case(
    caseid: int, 
    waveform_key: str, 
    substitutions_map: Dict[str, List[str]]
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    Loads a waveform for a given case, trying the primary key first, then substitutions.
    
    Args:
        caseid: VitalDB case ID
        waveform_key: The primary waveform identifier (e.g. 'SNUADC/ECG_II')
        substitutions_map: Dictionary mapping primary keys to lists of substitutes.
        
    Returns:
        wave: The loaded raw waveform (numpy array) or None
        spec_key_used: The key in WAVEFORM_SPECS that was actually used (e.g. 'SNUADC/ECG_V5')
        error: Error message string if loading failed, else None
    """
    wave = None
    spec_key_loaded = None
    spec = None
    
    # Resolve the primary spec key from the input waveform string
    primary_spec_key = ID_TO_SPEC_KEY.get(waveform_key)
    if not primary_spec_key and waveform_key in WAVEFORM_SPECS:
        primary_spec_key = waveform_key

    # 1. Try loading the primary waveform key
    if primary_spec_key:
        try:
            spec = WAVEFORM_SPECS[primary_spec_key]
            # VitalDB load_case can return None or empty
            wave = vitaldb.load_case(caseid, spec['id'], interval=1/spec['native_sr']) 
            if wave is not None and wave.size > 0:
                spec_key_loaded = primary_spec_key
        except Exception:
            pass 

    # 2. If primary failed, try substitutions
    if wave is None or wave.size == 0:
        for sub_id in substitutions_map.get(waveform_key, []):
            # Resolve substitute spec key
            sub_spec_key = ID_TO_SPEC_KEY.get(sub_id)
            if not sub_spec_key and sub_id in WAVEFORM_SPECS:
                sub_spec_key = sub_id
            
            if sub_spec_key:
                try:
                    spec = WAVEFORM_SPECS[sub_spec_key]
                    wave = vitaldb.load_case(caseid, sub_spec_key, interval=1/spec['native_sr'])
                    if wave is not None and wave.size > 0:
                        spec_key_loaded = sub_spec_key
                        break 
                except Exception:
                    continue 

    # 3. Check if loading failed entirely
    if wave is None or wave.size == 0:
        return None, None, 'empty_signal_or_missing'
        
    return wave, spec_key_loaded, None
