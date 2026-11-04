from reporting.display_dictionary import load_display_dictionary


def test_direct_feature_lookup_includes_unit():
    names = load_display_dictionary()

    label = names.feature_label("preop_egfr_ckdepi_2021", include_unit=True)

    assert "Estimated GFR" in label
    assert "mL/min/1.73 m²" in label
    assert "%" in names.feature_label("preop_pt", include_unit=True)
    assert "mg/dL" in names.feature_label("preop_crp", include_unit=True)
    assert "performed" not in names.feature_label("preop_ecg").lower()


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


def test_derived_waveform_statistics_do_not_inherit_source_signal_units():
    names = load_display_dictionary()
    for feature in ("Primus_AWP_CO_f1ecac", "SNUADC_ECG_II_SB_BinaryStats_mean_longstretch1"):
        entry = names.feature_entry(feature)
        assert entry.label != feature
        assert entry.unit is None
        assert "hPa" not in names.feature_label(feature, include_unit=True)
        assert "mV" not in names.feature_label(feature, include_unit=True)
    assert names.waveforms["Primus/AWP"]["unit"] == "hPa"
