"""Identify verified finite fallbacks without changing raw Catch24 outputs.

Audited against the PyPI pycatch22 0.4.5 source archive. Ordinary zero outputs
remain valid unless the specific implemented failure condition is verified.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
from importlib.metadata import version
import math
from pathlib import Path

import catch22_C
import numpy as np


EMBEDDING = "CO_Embed2_Dist_tau_d_expfit_meandiff"
FLUCTUATIONS = {
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
}
SOURCE_URL = "https://files.pythonhosted.org/packages/84/9c/452f83ae4d0e1f0fc36317b6b15c7a07214a1cc7205e19b35d77253c6cad/pycatch22-0.4.5.tar.gz"
SOURCE_SHA256 = "7ec844c659f22bedc66847ac866ef2bd86ffbbd4d8114b5e97f699f20a6f9f81"
DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


@lru_cache(maxsize=1)
def _library():
    if version("pycatch22") != "0.4.5":
        raise ValueError("Finite-fallback audit requires pycatch22 0.4.5")
    library = ctypes.CDLL(catch22_C.__file__)
    library.zscore_norm2.argtypes = [DOUBLE_POINTER, ctypes.c_int, DOUBLE_POINTER]
    library.zscore_norm2.restype = None
    library.co_firstzero.argtypes = [DOUBLE_POINTER, ctypes.c_int, ctypes.c_int]
    library.co_firstzero.restype = ctypes.c_int
    library.stddev.argtypes = [DOUBLE_POINTER, ctypes.c_int]
    library.stddev.restype = ctypes.c_double
    library.num_bins_auto.argtypes = [DOUBLE_POINTER, ctypes.c_int]
    library.num_bins_auto.restype = ctypes.c_int
    return library


@lru_cache(maxsize=128)
def fluctuation_scale_count(size: int) -> int:
    """Positive C round(exp(log-spaced scales)); integer size/2 division."""
    if size < 2:
        raise ValueError("Window must have at least two samples")
    low, high = math.log(5), math.log(size // 2)
    step = (high - low) / 49
    return len({math.floor(math.exp(low + index * step) + 0.5) for index in range(50)})


def embedding_fallback_status(window) -> dict:
    """Replay only the source's degenerate-distance SD/bin guard."""
    signal = np.array(window, dtype=np.float64, order="C", copy=True)
    if signal.ndim != 1 or len(signal) < 3 or not np.isfinite(signal).all():
        raise ValueError("Expected a finite one-dimensional window")
    library = _library()
    pointer = signal.ctypes.data_as(DOUBLE_POINTER)
    if not math.isfinite(library.stddev(pointer, len(signal))) or library.stddev(pointer, len(signal)) <= 0:
        return {"verified": False, "unresolved": True, "reason": "normalization_unresolved"}
    z = np.empty_like(signal)
    library.zscore_norm2(pointer, len(signal), z.ctypes.data_as(DOUBLE_POINTER))
    if not np.isfinite(z).all():
        return {"verified": False, "unresolved": True, "reason": "normalization_unresolved"}
    tau = library.co_firstzero(z.ctypes.data_as(DOUBLE_POINTER), len(z), len(z))
    if tau > len(z) / 10:
        tau = len(z) // 10
    count = len(z) - tau - 1
    if tau < 1 or count < 2:
        return {"verified": False, "unresolved": True, "reason": "embedding_lag_unresolved"}
    left = z[1:count + 1] - z[:count]
    right = z[tau:tau + count] - z[tau + 1:tau + count + 1]
    distances = np.sqrt(left * left + right * right)
    distance_pointer = distances.ctypes.data_as(DOUBLE_POINTER)
    sd = library.stddev(distance_pointer, count)
    # Vectorized distance arithmetic can differ by a few ULPs from scalar C.
    # Borderline guards remain unresolved rather than being declared failures.
    margin = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(distances))))
    borderline = not math.isfinite(sd) or abs(sd - 0.001) <= margin
    verified = (not borderline and sd < 0.001
                and library.num_bins_auto(distance_pointer, count) == 0)
    return {"verified": bool(verified), "unresolved": bool(borderline),
            "distance_sd": sd, "distance_count": count, "tau": tau}


def verified_fallbacks(window, feature_values, feature_names) -> np.ndarray:
    """Return a mask of source-verified failure conditions, never imputed values."""
    _library()
    values = np.asarray(feature_values, dtype=float)
    if len(values) != len(feature_names):
        raise ValueError("Descriptor values and identities must align")
    result = np.zeros(len(values), dtype=bool)
    n_scales = fluctuation_scale_count(len(window))
    for index, name in enumerate(feature_names):
        if name == EMBEDDING and values[index] == 0:
            result[index] = embedding_fallback_status(window)["verified"]
        elif name in FLUCTUATIONS:
            result[index] = values[index] == (0 if n_scales < 12 else 1.0 / n_scales)
    return result


def library_audit_metadata() -> dict:
    _library()
    return {
        "pycatch22_version": version("pycatch22"), "source_archive_url": SOURCE_URL,
        "source_archive_sha256": SOURCE_SHA256,
        "extension_sha256": hashlib.sha256(Path(catch22_C.__file__).read_bytes()).hexdigest(),
        "verified_conditions": ["embedding_distance_sample_sd_lt_0.001_with_zero_output",
                                "fluctuation_insufficient_scales_or_finite_no_split_fallthrough"],
        "borderline_policy": "Distance SD within 64*eps*max(1,max_distance) of 0.001 remains unresolved; raw features preserved.",
        "scope": "Identifies specific finite fallbacks; does not certify all internal numerics, estimate accuracy or physiological validity.",
    }
