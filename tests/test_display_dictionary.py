from reporting.display_dictionary import load_display_dictionary


def test_direct_feature_lookup_includes_unit():
    names = load_display_dictionary()

    label = names.feature_label("preop_egfr_ckdepi_2021", include_unit=True)

    assert "Estimated GFR" in label
    assert "mL/min/1.73 m²" in label


def test_catch22_feature_label_parses_waveform_and_aggregate():
    names = load_display_dictionary()

    label = names.feature_label("SNUADC_PLETH_DN_HistogramMode_5_mean")

    assert label.startswith("Pleth — Histogram mode (5 bins)")
    assert "Mean across windows" in label


def test_feature_set_labels_merge_defaults():
    names = load_display_dictionary()
    merged = names.feature_set_labels({"custom_key": "Custom Feature Set"})

    assert merged["preop_only"] in {"Preoperative only", "Preoperative Only"}
    assert merged["custom_key"] == "Custom Feature Set"


def test_branch_label_supports_short_form():
    names = load_display_dictionary()

    assert names.branch_label("windowed", use_short=True) == "Windowed"
