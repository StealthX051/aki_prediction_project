"""Invalid inputs stay missing; extreme held-out values cannot set cleaning rules."""

import numpy as np
import pandas as pd
import pytest

from model_creation import validation
from model_creation.preprocessing import FoldPreprocessor
from model_creation.utils import compute_quantile_cuts_per_feature
from reporting.display_dictionary import load_display_dictionary
from reporting.missingness_table import compute_missingness_table
from reporting.preop_descriptives import _build_descriptive_table, _ecg_display
from data_preparation import step_01_cohort_construction as cohort
from data_preparation.cohort_scopes import select_cohort


@pytest.mark.parametrize("family", ["logistic_regression", "xgboost", "ebm"])
def test_registered_numeric_strings_and_infinities_use_one_validity_policy(family):
    train = pd.DataFrame({
        "age": ["40", "bad", "60", np.inf, -np.inf],
        "Primus_CO2_mean": ["1", "2", "bad", np.inf, -np.inf],
        "inpatient_preop": ["1", "0", None, "bad", np.inf],
        "sex": pd.Series([" F ", "M", None, "", "  "], dtype="string"),
    })
    prep, result = validation.fit_preprocessor(train, preserve_nan=True, model_type=family)
    assert prep.numeric_columns_ == ["age", "Primus_CO2_mean", "inpatient_preop"]
    assert prep.categorical_columns_ == ["sex"]
    assert prep.clip_bounds_["age"] == pytest.approx((40.1, 59.9))
    assert "sex_F" in result and "sex___missing__" in result
    if family == "logistic_regression":
        assert np.isfinite(result.to_numpy()).all()
        assert prep.numeric_impute_values_["age"] == 50
        assert result["age__missing"].tolist() == [0, 1, 0, 1, 1]
    else:
        assert result["age"].isna().tolist() == [False, True, False, True, True]
        assert result["Primus_CO2_mean"].isna().tolist() == [False, False, True, True, True]
    heldout = prep.transform(pd.DataFrame({
        "age": [np.inf], "Primus_CO2_mean": [-np.inf], "inpatient_preop": ["bad"], "sex": ["UNSEEN"],
    }))
    assert not np.isinf(heldout.to_numpy()).any()
    assert "sex_UNSEEN" not in heldout


def test_winsorization_is_training_only_and_leaves_waveform_descriptors_unclipped():
    train = pd.DataFrame({"preop_aptt": [20., 30., 40., 400., np.inf],
                          "Primus_AWP_mean": [1., 2., 3., 4., np.inf]})
    prep = FoldPreprocessor().fit(train)
    expected = np.quantile([20., 30., 40., 400.], [.005, .995])
    np.testing.assert_allclose(prep.clip_bounds_["preop_aptt"], expected)
    assert "Primus_AWP_mean" not in prep.clip_bounds_
    before = dict(prep.clip_bounds_)
    heldout = prep.transform(pd.DataFrame({"preop_aptt": [-1000., 10000.],
                                          "Primus_AWP_mean": [-1000., 10000.]}))
    np.testing.assert_allclose(heldout.preop_aptt, expected)
    assert heldout.Primus_AWP_mean.tolist() == [-1000., 10000.]
    assert prep.clip_bounds_ == before
    assert train.loc[3, "preop_aptt"] == 400  # raw descriptives remain intact


def test_rare_department_does_not_merge_missing_with_observed_categories():
    prep = FoldPreprocessor(rare_category_min_count=2)
    result = prep.fit_transform(pd.DataFrame({"department": ["A", "A", "rare", None]}))
    assert prep.rare_mappings_["department"] == {"rare"}
    assert result.loc[3, "department___missing__"] == 1
    assert result.loc[3, "department___other__"] == 0


def test_ebm_cutpoints_exclude_nonfinite_inputs_and_retain_empty_columns():
    features = np.array([[1., np.nan], [3., np.inf], [np.inf, -np.inf], [-np.inf, np.nan]])
    cuts = compute_quantile_cuts_per_feature(features, max_bins=4)
    np.testing.assert_allclose(cuts[0], [5 / 3, 7 / 3])
    assert cuts[1] == "continuous"


def test_missingness_counts_unusable_numeric_values_and_keeps_unknown_categories():
    frame = pd.DataFrame({"age": ["40", "bad", np.inf, -np.inf, None],
                          "Primus_CO2_mean": [1., 2., np.nan, np.inf, 3.],
                          "department": ["A", "Unknown", "", "  ", None]})
    result = compute_missingness_table(frame, load_display_dictionary()).set_index("feature_name")
    assert result.loc["age", "missing_count"] == 4
    assert result.loc["Primus_CO2_mean", "missing_count"] == 2
    assert result.loc["department", "missing_count"] == 3
    assert set(result.n_operations) == {5}


def test_table_one_uses_same_blank_and_unusable_value_denominators():
    frame = pd.DataFrame({"age": [40., np.inf, "bad", None],
                          "department": ["A", "", "  ", None]})
    table = _build_descriptive_table(frame, None, continuous_features=["age"],
                                    categorical_features=["department"],
                                    alpha=.05, max_sample=5000, random_state=42,
                                    display_dictionary=load_display_dictionary())
    # Check both feature rows independently: one correct row must not hide
    # an incorrect denominator or missing-value policy in the other.
    rows = table.loc[table.Type.isin(["Continuous", "Categorical"])]
    assert rows["Missing n (%)"].tolist() == ["3 (75.0%)", "3 (75.0%)"]
    assert rows.loc[rows.Type.eq("Continuous"), "Summary"].tolist() == ["40.00 (40.00, 40.00)"]
    assert table.loc[table.Feature.eq("  Missing"), "Summary"].tolist() == ["3 (75.0%)"]


def test_cohort_category_normalization_preserves_missing_asa_and_source_values():
    frame = pd.DataFrame({"caseid": range(4), "subjectid": range(4),
                          "asa": [" v ", "vi", "II", None],
                          "ane_type": ["General", "General", " general ", "GENERAL"],
                          "eligible_any_aki": [1] * 4, "aki_label": [0] * 4,
                          "eligible_icu_admission": [1] * 4, "y_icu_admit": [0] * 4})
    selected = cohort._restrict_to_general_anesthesia(cohort._exclude_asa_v_vi(frame))
    assert selected.index.tolist() == [2, 3]
    assert selected.loc[2, "ane_type"] == " general "
    assert pd.isna(selected.loc[3, "asa"])
    for scope in ("main_shared", "icu_waveform_sensitivity"):
        scoped = select_cohort(selected, scope)
        pd.testing.assert_frame_equal(scoped, selected)


def test_numeric_categorical_infinities_are_missing_across_all_consumers():
    frame = pd.DataFrame({"asa": [2., np.inf, -np.inf, np.nan]})
    result = FoldPreprocessor().fit_transform(frame)
    assert result["asa___missing__"].tolist() == [0, 1, 1, 1]
    missingness = compute_missingness_table(frame, load_display_dictionary())
    assert missingness.loc[0, "missing_count"] == 3
    table = _build_descriptive_table(frame, None, continuous_features=[],
                                    categorical_features=["asa"],
                                    alpha=.05, max_sample=5000, random_state=42,
                                    display_dictionary=load_display_dictionary())
    assert "3 (75.0%)" in table["Missing n (%)"].tolist()
    missing = table.loc[table.Feature.eq("  Missing"), "Summary"]
    assert missing.tolist() == ["3 (75.0%)"]
    assert _ecg_display(pd.Series([np.inf, -np.inf])).tolist() == ["Missing", "Missing"]
