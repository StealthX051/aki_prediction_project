"""Tests for CKD-EPI 2021 eGFR derivation."""

import numpy as np
import pandas as pd
import pytest

from data_preparation.step_03_preop_prep import add_derived_preop_features, build_preop_labs_from_labdata


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
                preop_cr=100 / 88.4,  # reference example explicitly expressed in source mg/dL
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
    assert result["preop_cr"].isna().all()


def test_high_source_creatinine_is_not_reinterpreted_as_micromoles():
    rows = [_base_row(preop_cr=25.62, sex="M", age=60), _base_row(preop_cr=np.inf)]
    result = add_derived_preop_features(pd.DataFrame(rows))
    expected = 142 * (25.62 / 0.9) ** -1.2 * 0.9938 ** 60
    np.testing.assert_allclose(result.loc[0, "preop_egfr_ckdepi_2021"], expected)
    assert result.loc[0, "preop_egfr_ckdepi_2021"] < 5
    assert pd.isna(result.loc[1, "preop_egfr_ckdepi_2021"])


@pytest.mark.parametrize("adm", [None, pd.NA, "unknown", np.inf, -np.inf])
def test_unknown_admission_time_preserves_missing_inpatient_flag(adm):
    result = add_derived_preop_features(pd.DataFrame([_base_row(adm=adm)]))
    assert pd.isna(result.loc[0, "inpatient_preop"])


def test_known_admission_times_preserve_existing_inpatient_definition():
    result = add_derived_preop_features(pd.DataFrame([_base_row(adm=value) for value in [-1, 0, 1, "-10"]]))
    assert result["inpatient_preop"].tolist() == [1, 0, 0, 1]


@pytest.mark.parametrize("sex", [None, pd.NA, "", "unknown", "other"])
def test_egfr_does_not_assume_male_for_unknown_sex(sex):
    result = add_derived_preop_features(pd.DataFrame([_base_row(sex=sex)]))
    assert pd.isna(result.loc[0, "preop_egfr_ckdepi_2021"])


@pytest.mark.parametrize("age", [None, pd.NA, "unknown", np.inf, -np.inf])
def test_egfr_requires_finite_numeric_age(age):
    result = add_derived_preop_features(pd.DataFrame([_base_row(age=age)]))
    assert pd.isna(result.loc[0, "preop_egfr_ckdepi_2021"])


def test_egfr_normalizes_derivation_sex_without_changing_source_categories():
    result = add_derived_preop_features(pd.DataFrame([
        _base_row(sex=" f "), _base_row(sex=" m ", age=60, preop_cr=100 / 88.4),
    ]))
    assert result["sex"].tolist() == [" f ", " m "]
    np.testing.assert_allclose(result.loc[0, "preop_egfr_ckdepi_2021"], 105.297601, rtol=1e-6)
    np.testing.assert_allclose(result.loc[1, "preop_egfr_ckdepi_2021"], 74.312457, rtol=1e-6)


def test_preop_labs_select_latest_finite_result_with_valid_preoperative_time():
    labs = pd.DataFrame({
        "caseid": [1] * 8 + [2, 3, 3],
        "name": ["wbc"] * 8 + ["crp", "lac", "lac"],
        "dt": [-20, "-10", -9, -8, "unknown", np.inf, -np.inf, 0, -5, -30 * 86400, -30 * 86400 - 1],
        "result": [1, "2", np.inf, "invalid", 3, 4, 5, 6, 0, 7, 8],
    })
    original = labs.copy(deep=True)
    result = build_preop_labs_from_labdata(labs).set_index("caseid")
    assert result.loc[1, "preop_wbc"] == 2
    assert result.loc[2, "preop_crp"] == 0
    assert result.loc[3, "preop_lac"] == 7
    assert pd.isna(result.loc[1, "preop_lac"])
    pd.testing.assert_frame_equal(labs, original)
