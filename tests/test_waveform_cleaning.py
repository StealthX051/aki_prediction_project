"""Synthetic numerical validity checks; never load research waveforms."""

import numpy as np
import pytest
import pandas as pd

from data_preparation import step_02_catch_22 as extraction
from data_preparation import waveform_processing as signals


@pytest.fixture
def synthetic_extraction(monkeypatch):
    spec = {"native_sr": 100, "target_sr": 100, "filter_type": "lowpass",
            "filter_order": 4, "filter_band": 10}
    monkeypatch.setitem(extraction.WAVEFORM_SPECS, "synthetic", spec)
    monkeypatch.setattr(extraction, "GENERATE_FULL_FEATURES", True)
    monkeypatch.setattr(extraction, "GENERATE_WINDOWED_FEATURES", False)
    monkeypatch.setattr(extraction.pycatch22, "catch22_all",
                        lambda values, catch24: {"names": ["feature"], "values": [float(np.mean(values))]})
    return spec


def _supply_wave(monkeypatch, wave):
    monkeypatch.setattr(extraction, "load_and_validate_case",
                        lambda *args: (wave.copy(), "synthetic", None))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("count,accepted", [(50, True), (51, False)])
def test_all_nonfinite_samples_share_exact_five_percent_rule(monkeypatch, synthetic_extraction, invalid, count, accepted):
    wave = np.sin(np.arange(1000) / 10)
    wave[:count] = invalid
    _supply_wave(monkeypatch, wave)
    seen = []
    monkeypatch.setattr(extraction, "process_signal", lambda values, *args: (seen.append(values) or values, None))
    result, _ = extraction._process_case((1, 0, 10, "synthetic"))
    if accepted:
        assert "error" not in result and "error" not in result["full"]
        assert np.isfinite(seen[0]).all()
        # Preserve the established nearest-value fill at the leading boundary.
        np.testing.assert_array_equal(seen[0][:count], np.repeat(wave[count], count))
    else:
        assert result["error"]["error"] == "invalid_signal_gt_5_pct_nonfinite"
        assert not seen


def test_empty_operation_slice_is_rejected_before_interpolation(monkeypatch, synthetic_extraction):
    _supply_wave(monkeypatch, np.arange(1000, dtype=float))
    result, _ = extraction._process_case((1, 10, 10, "synthetic"))
    assert result["error"]["error"] == "invalid_signal_empty"


@pytest.mark.parametrize("values", [np.array([]), np.array([1., np.inf]), np.array([np.nan])])
def test_signal_processing_rejects_invalid_input(synthetic_extraction, values):
    processed, error = signals.process_signal(values, synthetic_extraction, 1, "synthetic")
    assert processed.size == 0 and error == "invalid_signal_empty_or_nonfinite"


def test_filter_failure_to_produce_finite_values_is_visible(monkeypatch, synthetic_extraction):
    monkeypatch.setattr(signals.scipy.signal, "sosfiltfilt", lambda *args: np.array([np.nan]))
    processed, error = signals.process_signal(np.ones(100), synthetic_extraction, 1, "synthetic")
    assert processed.size == 0 and error == "invalid_filtered_signal_empty_or_nonfinite"


@pytest.mark.parametrize("values", [np.array([]), np.array([np.inf])])
def test_resampling_failure_to_produce_finite_values_is_visible(monkeypatch, synthetic_extraction, values):
    spec = {**synthetic_extraction, "target_sr": 10}
    monkeypatch.setattr(signals, "harmonize_sr", lambda *args: values)
    processed, error = signals.process_signal(np.sin(np.arange(1000) / 10), spec, 1, "synthetic")
    assert processed.size == 0 and error == "invalid_resampled_signal_empty_or_nonfinite"


@pytest.mark.parametrize("values", [np.array([]), np.array([np.inf])])
def test_full_feature_resampling_is_also_validated(monkeypatch, synthetic_extraction, values):
    _supply_wave(monkeypatch, np.sin(np.arange(1000) / 10))
    monkeypatch.setattr(extraction, "harmonize_sr", lambda *args: values)
    result, _ = extraction._process_case((1, 0, 10, "synthetic"))
    assert result["full"]["error"] == "invalid_full_resampled_signal_empty_or_nonfinite"


def test_full_descriptors_canonicalize_nonfinite_values_without_dropping_channel(monkeypatch, synthetic_extraction):
    _supply_wave(monkeypatch, np.sin(np.arange(1000) / 10))
    monkeypatch.setattr(extraction.pycatch22, "catch22_all", lambda *args, **kwargs:
                        {"names": ["finite", "nan", "positive_inf", "negative_inf"],
                         "values": [2., np.nan, np.inf, -np.inf]})
    result, _ = extraction._process_case((1, 0, 10, "synthetic"))
    assert result["full"]["finite"] == 2
    assert all(np.isnan(result["full"][name]) for name in ("nan", "positive_inf", "negative_inf"))


def test_window_aggregation_uses_float64_and_preserves_undefined_propagation(monkeypatch, synthetic_extraction):
    _supply_wave(monkeypatch, np.sin(np.arange(1500) / 10))
    monkeypatch.setattr(extraction, "GENERATE_FULL_FEATURES", False)
    monkeypatch.setattr(extraction, "GENERATE_WINDOWED_FEATURES", True)
    monkeypatch.setattr(extraction, "WIN_SEC", 10)
    monkeypatch.setattr(extraction, "SLIDE_SEC", 5)
    features = iter(([1., np.inf, 1e30], [3., 5., 2e30]))
    monkeypatch.setattr(extraction.pycatch22, "catch22_all", lambda *args, **kwargs:
                        {"names": ["finite", "undefined", "large"], "values": next(features)})
    result, _ = extraction._process_case((1, 0, 15, "synthetic"))
    row = result["windowed"]
    assert row["finite_mean"] == 2
    assert row["finite_std"] == pytest.approx(np.sqrt(2))
    assert all(np.isnan(row[f"undefined_{suffix}"]) for suffix in ("mean", "std", "min", "max"))
    assert row["large_std"] == pytest.approx(1e30 / np.sqrt(2))
    assert row["large_mean"] == pytest.approx(1.5e30)


def test_one_valid_window_keeps_sample_sd_missing(monkeypatch, synthetic_extraction):
    _supply_wave(monkeypatch, np.sin(np.arange(1000) / 10))
    monkeypatch.setattr(extraction, "GENERATE_FULL_FEATURES", False)
    monkeypatch.setattr(extraction, "GENERATE_WINDOWED_FEATURES", True)
    monkeypatch.setattr(extraction, "WIN_SEC", 10)
    monkeypatch.setattr(extraction, "SLIDE_SEC", 5)
    with pytest.warns(RuntimeWarning, match="Degrees of freedom"):
        result, _ = extraction._process_case((1, 0, 10, "synthetic"))
    assert np.isnan(result["windowed"]["feature_std"])
    assert np.isfinite(result["windowed"]["feature_mean"])


@pytest.mark.parametrize("source_sr,target_sr,rejected_hz,alias_hz", [
    (500, 100, 120, 20), (100, 10, 17, 3), (62.5, 10, 17, 3),
])
def test_resampling_preserves_passband_and_suppresses_aliases(source_sr, target_sr, rejected_hz, alias_hz):
    times = np.arange(int(20 * source_sr)) / source_sr
    signal = 3 + np.sin(2 * np.pi * times) + np.sin(2 * np.pi * rejected_hz * times)
    original = signal.copy()
    result = signals.harmonize_sr(signal, source_sr, target_sr)
    output_times = np.arange(result.size) / target_sr
    interior = (output_times >= 2) & (output_times < 18)
    centered = result[interior] - 3
    def amplitude(hz):
        return 2 * abs(np.mean(centered * np.exp(-2j * np.pi * hz * output_times[interior])))
    assert amplitude(1) == pytest.approx(1, abs=.005)
    assert amplitude(alias_hz) < .003  # <0.3% of unit-amplitude rejected tone
    assert np.mean(result[interior]) == pytest.approx(3, abs=.001)
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("source_sr,target_sr", [(500, 100), (100, 10), (62.5, 10)])
def test_resampling_is_centered_without_impulse_delay(source_sr, target_sr):
    signal = np.zeros(int(4 * source_sr))
    signal[int(2 * source_sr)] = 1
    result = signals.harmonize_sr(signal, source_sr, target_sr)
    assert np.argmax(result) == int(2 * target_sr)


@pytest.mark.parametrize("source_sr,target_sr,length", [
    # Empty, singleton, and shortest line-extension inputs need one check each.
    (500, 100, 0), (500, 100, 1), (500, 100, 2),
    # Output-length transitions differ for each integer decimation factor.
    (500, 100, 4), (500, 100, 5), (500, 100, 6),
    (100, 10, 9), (100, 10, 10), (100, 10, 11),
    # Fractional sampling must round on the physical grid, including long inputs.
    (62.5, 10, 6), (62.5, 10, 7),
    (62.5, 10, 61), (62.5, 10, 62), (62.5, 10, 63),
    (62.5, 10, 624), (62.5, 10, 625), (62.5, 10, 626),
])
def test_resampling_length_short_inputs_and_constant_boundaries(source_sr, target_sr, length):
    signal = np.full(length, 7.)
    result = signals.harmonize_sr(signal, source_sr, target_sr)
    assert result.size == int(np.ceil(length * target_sr / source_sr))
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, 7., rtol=.001, atol=1e-8)


def test_fractional_resampling_uses_physical_time_grid_instead_of_stretching_endpoints():
    source_sr, target_sr = 62.5, 10
    # An exactly linear signal also checks the explicit boundary extension.
    signal = np.arange(626) / source_sr
    result = signals.harmonize_sr(signal, source_sr, target_sr)
    expected = np.arange(result.size) / target_sr
    np.testing.assert_allclose(result, expected, rtol=.001, atol=.002)


def test_same_rate_leaves_values_unchanged_without_filtering():
    signal = np.array([3., -2., 8., 1.])
    np.testing.assert_array_equal(signals.harmonize_sr(signal, 62.5, 62.5), signal)


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_resampling_rejects_invalid_rates(rate):
    for source_sr, target_sr in [(rate, 10), (100, rate)]:
        with pytest.raises(ValueError, match="finite and positive"):
            signals.harmonize_sr(np.ones(100), source_sr, target_sr)


@pytest.mark.parametrize("signal", [np.ones((2, 2)), np.array([1., np.nan]), np.array([1., np.inf])])
def test_resampling_rejects_nonfinite_or_nonvector_inputs(signal):
    with pytest.raises(ValueError, match="finite one-dimensional"):
        signals.harmonize_sr(signal, 100, 10)


@pytest.mark.parametrize("channel", list(signals.WAVEFORM_SPECS))
def test_actual_filter_resampling_and_catch24_paths_stay_operation_local(monkeypatch, channel):
    spec = signals.WAVEFORM_SPECS[channel]
    source_sr = spec["native_sr"]
    times = np.arange(int(32 * source_sr)) / source_sr
    wave = 3 + np.sin(2 * np.pi * times) + .2 * np.sin(2 * np.pi * 17 * times)
    monkeypatch.setattr(extraction, "GENERATE_FULL_FEATURES", True)
    monkeypatch.setattr(extraction, "GENERATE_WINDOWED_FEATURES", True)
    original_catch24 = extraction.pycatch22.catch22_all
    sample_counts = []
    def track_samples(values, **kwargs):
        sample_counts.append(len(values))
        return original_catch24(values, **kwargs)
    monkeypatch.setattr(extraction.pycatch22, "catch22_all", track_samples)
    monkeypatch.setattr(extraction, "load_and_validate_case", lambda *args: (wave.copy(), channel, None))
    original, _ = extraction._process_case((1, 1, 31, channel))
    assert sample_counts[0] == 30 * 10  # full-operation features retain 10 Hz
    assert sample_counts[1:] == [int(10 * spec["target_sr"])] * 5
    for branch, expected_features in [("full", 24), ("windowed", 96)]:
        row = original[branch]
        assert "error" not in row and len(row) == expected_features + 2
        assert row["caseid"] == 1 and row["waveform"] == channel
        values = [v for name, v in row.items() if name not in {"caseid", "waveform"}]
        assert not np.isinf(values).any()
        assert np.isfinite(values).any()
    # Outside-operation values must not affect filtering, padding, or features.
    wave[:int(source_sr)] = 1e6
    wave[int(31 * source_sr):] = -1e6
    changed, _ = extraction._process_case((1, 1, 31, channel))
    for branch in ("full", "windowed"):
        for feature, value in original[branch].items():
            if feature not in {"caseid", "waveform"}:
                np.testing.assert_allclose(changed[branch][feature], value, rtol=0, atol=0, equal_nan=True)


def test_extraction_fingerprint_records_resampling_and_rejects_changed_policy(tmp_path):
    from data_preparation.artifact_metadata import ArtifactCompatibilityError
    cohort = pd.DataFrame({"caseid": [1], "opstart": [0], "opend": [30]})
    fingerprint = extraction.build_extraction_fingerprint(cohort, "full")
    assert fingerprint["scipy_version"]
    assert fingerprint["extraction_protocol_version"] == 3
    assert fingerprint["full_target_sr"] == 10
    assert fingerprint["resampling_policy"]["window"] == ["kaiser", 5.0]
    output, errors = tmp_path / "features.csv", tmp_path / "errors.csv"
    conf = {"file": output, "error_file": errors,
            "processed_pairs": {(1, "Primus/CO2")},
            "features": [{"caseid": 1, "waveform": "Primus/CO2", "feature": 1.}], "errors": []}
    extraction._save_mode_checkpoint(conf, fingerprint)
    assert extraction.load_processed_pairs(output, errors, fingerprint) == {(1, "Primus/CO2")}
    for previous_protocol in (1, 2):
        with pytest.raises(ArtifactCompatibilityError, match="fingerprint mismatch"):
            extraction.load_processed_pairs(output, errors,
                {**fingerprint, "extraction_protocol_version": previous_protocol})
    with pytest.raises(ArtifactCompatibilityError, match="fingerprint mismatch"):
        extraction.load_processed_pairs(output, errors,
            {**fingerprint, "resampling_policy": {**fingerprint["resampling_policy"], "padtype": "constant"}})
