"""Tests for CKD-EPI 2021 eGFR derivation."""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data_preparation.step_03_preop_prep import add_derived_preop_features


def _base_row(**overrides):
    base = {
        "adm": -1000,
        "sex": "F",
        "age": 50,
        "preop_cr": 0.7,
        "preop_bun": 20,
        "preop_alb": 4.0,
        "preop_hb": 13.5,
        "preop_na": 140,
        "preop_hco3": 24,
        "preop_be": 0,
        "preop_pao2": 100,
        "preop_paco2": 40,
        "preop_sao2": 98,
    }

    base.update(overrides)
    return base


def test_ckdepi_2021_matches_reference_values():
    """Regression check using NKF CKD-EPI 2021 calculator examples."""

    df = pd.DataFrame(
        [
            _base_row(),
            _base_row(
                sex="M",
                age=60,
                # 100 µmol/L -> 1.1312 mg/dL after conversion
                preop_cr=100,
            ),
        ]
    )

    result = add_derived_preop_features(df.copy())

    expected = np.array([105.297601, 74.312457])
    np.testing.assert_allclose(result["preop_egfr_ckdepi_2021"], expected, rtol=1e-6)


def test_nonpositive_creatinine_yields_nan_before_imputation():
    rows = [
        _base_row(preop_cr=0),
        _base_row(preop_cr=-1, sex="M"),
    ]

    df = pd.DataFrame(rows)
    result = add_derived_preop_features(df.copy())

    assert result["preop_egfr_ckdepi_2021"].isna().all()
