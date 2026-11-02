"""Window audit preserves production grids and undefined-value propagation."""
import numpy as np
import pytest

from scripts import window_duration_pilot as pilot


def test_grid_rounding_flat_windows_and_nan_propagation(monkeypatch):
    def features(window, catch24):
        values = np.arange(24, dtype=float) + window.mean()
        if window[0] == 0:
            values[4] = np.nan
        return {"names": pilot.FEATURE_NAMES, "values": values}
    monkeypatch.setattr(pilot.pycatch22, "catch22_all", features)
    signal = np.arange(1300, dtype=float)
    result = pilot.extract_grid(signal, 62.5, 10)
    assert result["candidate_starts"].tolist() == [0, 312, 624]
    assert result["starts"].tolist() == [0, 312, 624]
    assert np.isnan(result["aggregates"][:, 4]).all()
    expected_means = np.array([312, 624, 936], dtype=float)
    assert result["aggregates"][:, 0] == pytest.approx([
        expected_means.mean(), expected_means.std(ddof=1), 312, 936])
    assert pilot.extract_grid(np.zeros(1300), 62.5, 10)["values"].shape == (0, 24)


def test_terminal_partial_and_single_window_sample_sd():
    values = np.arange(24, dtype=float).reshape(1, 24)
    result = pilot.aggregate(values)
    assert np.isnan(result[1]).all()
    assert np.array_equal(result[0], values[0])
    signal = np.sin(np.arange(700) / 12)
    grid = pilot.extract_grid(signal, 62.5, 10, 78)
    assert len(grid["starts"]) == 0  # 78 + 625 exceeds 700; no padded window.


def test_grid_rejects_unprocessed_nonfinite_signal():
    with pytest.raises(ValueError, match="finite"):
        pilot.extract_grid(np.array([1., np.inf]), 100, 10)
