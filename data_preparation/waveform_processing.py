import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
import math
import logging
from typing import Tuple, Dict, Any, Optional

# === Signal Processing Helpers ===
# Note: Using Butterworth, order 4 (unless specified), zero-phase with sosfiltfilt
# to preserve morphology and ensure numerical stability at low cutoffs.

def bpf(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """Zero-phase band-pass filter using Second-Order Sections (SOS)"""
    nyquist = fs / 2
    sos = butter(order, [lo / nyquist, hi / nyquist], btype='band', output='sos')
    return sosfiltfilt(sos, x)

def lpf(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass filter using Second-Order Sections (SOS)"""
    nyquist = fs / 2
    sos = butter(order, fc / nyquist, btype='low', output='sos')
    return sosfiltfilt(sos, x)

def hpf(x: np.ndarray, fs: float, fc: float, order: int = 2) -> np.ndarray:
    """Zero-phase high-pass filter using Second-Order Sections (SOS)"""
    nyquist = fs / 2
    sos = butter(order, fc / nyquist, btype='high', output='sos')
    return sosfiltfilt(sos, x)

def aa_downsample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Anti-aliased resampling using polyphase filtering"""
    return resample_poly(x, up, down)

def harmonize_sr(x: np.ndarray, src_sr: float, dst_sr: float) -> np.ndarray:
    """Resample x from src_sr to dst_sr using polyphase (resample_poly)."""
    if src_sr == dst_sr:
        return x
    
    # Calculate rational fraction for resampling
    g = math.gcd(int(src_sr * 100), int(dst_sr * 100))
    up = int(dst_sr * 100) // g
    down = int(src_sr * 100) // g
    
    return aa_downsample(x, up, down)

# === Waveform Processing Specifications ===
# Maps keys (used in inputs.py) to their VitalDB track IDs, native SR,
# target SR (after resampling), and filtering parameters.
WAVEFORM_SPECS = {
    # 500 Hz signals -> 100 Hz
    'ECG_II': {'id': 'SNUADC/ECG_II', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.5, 40)},  # 0.5–40 Hz morphology for ML; cf. diagnostic 0.05–150 (Kligfield 2007, Circulation); QRS 5–15 (Pan & Tompkins 1985, IEEE TBME)
    'ECG_V5': {'id': 'SNUADC/ECG_V5', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.5, 40)},  # same (Kligfield 2007; Pan & Tompkins 1985)

    'PLETH':  {'id': 'SNUADC/PLETH', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.1, 10)},  # 0.1–10 Hz preserves notch/phase; <10 dampens notch (Lapitan 2024, Sci Rep; Park 2022, Front Physiol)

    'ART':    {'id': 'SNUADC/ART', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.3, 30)},  # ABP morphology incl. dicrotic notch; arterial content to ~20–25 Hz (Watanabe 2020, J Anesth; Pal 2024, CMPB)
    'FEM':    {'id': 'SNUADC/FEM', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.3, 30)},  # same as ART (Watanabe 2020; Pal 2024)
    'CVP':    {'id': 'SNUADC/CVP', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.05, 10)}, # retain respiratory (~0.2–0.3 Hz) + cardiac (~1–2 Hz) components (Magder 2015, Curr Opin Crit Care)

    # 62.5 Hz signals -> 62.5 Hz (no resampling)
    'AWP':    {'id': 'Primus/AWP', 'native_sr': 62.5, 'target_sr': 62.5, 'up': 1, 'down': 1,
               'filter': ('lpf', 12)},       # LPF ~10–15 Hz keeps breath shape/PEEP (de Haro 2024, Crit Care; Thome 1998, J Appl Physiol)
    'CO2':    {'id': 'Primus/CO2', 'native_sr': 62.5, 'target_sr': 62.5, 'up': 1, 'down': 1,
               'filter': ('lpf', 8)},        # Morphology LPF 5–10 Hz; for ventilation detection use ≤1–2 Hz (Gutiérrez 2018, PLoS One; Leturiondo 2017, CinC)

    # 128 Hz signals -> 128 Hz (no resampling)
    'EEG1':   {'id': 'BIS/EEG1_WAV', 'native_sr': 128, 'target_sr': 128, 'up': 1, 'down': 1,
               'filter': ('bpf', 0.5, 30)},  # anesthesia EEG preproc 0.5–30/40 Hz is common (Schmidlin 2001, Br J Anaesth; Nagaraj 2018, PLoS One)
    'EEG2':   {'id': 'BIS/EEG2_WAV', 'native_sr': 128, 'target_sr': 128, 'up': 1, 'down': 1,
               'filter': ('bpf', 0.5, 30)},  # same (Schmidlin 2001; Nagaraj 2018)

    # 180 Hz signals -> 90 Hz
    'ABP':    {'id': 'CardioQ/ABP', 'native_sr': 180, 'target_sr': 90, 'up': 1, 'down': 2,
               'filter': ('bpf', 0.3, 30)},  # as invasive ABP; retain notch/harmonics (Watanabe 2020, J Anesth; Lam 2021, Cureus)
    'FLOW':   {'id': 'CardioQ/FLOW', 'native_sr': 180, 'target_sr': 90, 'up': 1, 'down': 2,
               'filter': ('bpf', 0.5, 20)},  # esophageal Doppler flow: suppress drift, keep systolic accel/FTc features (Deltex 2018, Operating Handbook)
}

def process_signal(seg: np.ndarray, spec: Dict[str, Any], caseid: int, waveform_key: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Applies filtering and resampling to a raw signal segment based on the specification.
    Returns (processed_signal, error_message).
    """
    native_sr = spec['native_sr']
    
    # 1. Filtering (at native SR)
    try:
        filt_type, *params = spec['filter']
        if filt_type == 'bpf':
            seg = bpf(seg, fs=native_sr, lo=params[0], hi=params[1])
        elif filt_type == 'lpf':
            seg = lpf(seg, fs=native_sr, fc=params[0])
        elif filt_type == 'hpf':
            seg = hpf(seg, fs=native_sr, fc=params[0])
    except ValueError as e:
        # Handle signals too short for filtering
        msg = f"Case {caseid} {waveform_key}: Filtering error (signal likely too short). {e}"
        logging.warning(msg)
        return None, f'filtering_error_signal_too_short'
    except Exception as e:
        msg = f"Case {caseid} {waveform_key}: Unknown filtering error. {e}"
        logging.error(msg)
        return None, f'filtering_error_unknown'

    # 2. Anti-alias downsampling
    if spec['up'] != 1 or spec['down'] != 1:
        try:
            seg = aa_downsample(seg, spec['up'], spec['down'])
        except Exception as e:
            msg = f"Case {caseid} {waveform_key}: Resampling error. {e}"
            logging.error(msg)
            return None, f'resampling_error_{e}'
            
    return seg, None
