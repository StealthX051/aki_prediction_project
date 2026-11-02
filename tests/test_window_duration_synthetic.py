"""Synthetic challenge checks do not fit estimators or use clinical data."""
import numpy as np
import pytest
import json
import sys
from types import SimpleNamespace

from scripts import window_duration_synthetic as synthetic
from scripts.window_duration_synthetic import ar_process, episode_envelope, generate_pair, summarize_pair


def test_frequency_change_is_continuous_and_noise_is_paired():
    base, changed, metadata = generate_pair(100, "sine_0p2", 0, 2, duration=80, episode=20)
    assert np.max(np.abs(np.diff(changed))) <= 2 * np.pi * 0.25 / 100 + 1e-10
    before = int(metadata["episode_start_seconds"] * 100)
    np.testing.assert_allclose(base[:before], changed[:before], atol=1e-11)
    noisy_base, noisy_changed, _ = generate_pair(100, "sine_0p2", 0.3, 2, duration=80, episode=20)
    np.testing.assert_allclose(noisy_changed - noisy_base, changed - base, atol=1e-12)
    times = np.arange(0, 80, 0.01)
    envelope = episode_envelope(times, 30, 50, 5)
    assert envelope[3000] == envelope[5000] == 0
    assert envelope[3500] == envelope[4500] == 1


@pytest.mark.parametrize("coefficient", [0.5, 0.9])
def test_ar_change_keeps_variance_scale(coefficient):
    innovations = np.random.default_rng(41).standard_normal(100_000)
    values = ar_process(innovations, np.full(len(innovations), coefficient))[1000:]
    assert np.var(values) == pytest.approx(1, abs=0.04)
    assert np.corrcoef(values[:-1], values[1:])[0, 1] == pytest.approx(coefficient, abs=0.015)


def test_long_windows_report_absent_episode_support_and_nan_propagation():
    base = {"starts": np.array([0, 32000]), "feature_names": ["a", "b"],
            "values": np.array([[1., np.nan], [2., 3.]]), "aggregates": np.zeros((4, 2))}
    changed = {**base, "aggregates": np.ones((4, 2))}
    metadata = {"episode_start_seconds": 570, "episode_end_seconds": 630}
    result = summarize_pair(base, changed, 640, 100, metadata)
    assert result["episode_contained_windows"] == 0
    assert result["episode_centre_windows"] == 0
    assert result["outside_windows"] == 0
    assert result["crossing_windows"] == 2
    assert np.isnan(result["episode_paired_mean_delta"]).all()
    assert np.isnan(result["baseline_outside_sd"]).all()
    np.testing.assert_equal(result["aggregate_delta"], np.ones((4, 2)))


def test_completed_trace_resume_checks_hashes_before_skipping(tmp_path, monkeypatch):
    stem = synthetic.trace_stem(100, "sine_1", 0.1, 0)
    artifacts = {}
    for suffix in (".csv", ".npz"):
        path = tmp_path / f"{stem}{suffix}"
        path.write_bytes(b"synthetic checkpoint")
        artifacts[path.name] = synthetic.sha256(path)
    (tmp_path / f"{stem}.json").write_text(json.dumps({
        "protocol_sha256": "frozen", "artifact_sha256": artifacts,
        "feature_rows": 168, "window_counts": {}}))
    monkeypatch.setattr(synthetic, "generate_pair", lambda *a, **k: pytest.fail("Completed trace regenerated"))
    assert synthetic.run_trace((100, "sine_1", 0.1, 0, tmp_path, "frozen"))["reused"]
    with pytest.raises(ValueError, match="protocol changed"):
        synthetic.verified_trace(tmp_path, stem, "new-code")
    damaged = tmp_path / f"{stem}.npz"
    damaged.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="artifact changed"):
        synthetic.run_trace((100, "sine_1", 0.1, 0, tmp_path, "frozen"))
    assert damaged.read_bytes() == b"damaged"


def test_interrupted_trace_replaces_only_its_own_unpublished_files(tmp_path, monkeypatch):
    stem = synthetic.trace_stem(10, "sine_1", 0.1, 0)
    for suffix in (".npz", ".csv.partial", ".json.partial"):
        (tmp_path / f"{stem}{suffix}").write_bytes(b"interrupted")
    unrelated = tmp_path / "other-trace.npz.partial"
    unrelated.write_bytes(b"keep me")
    metadata = {"episode_start_seconds": 10, "episode_end_seconds": 20}
    monkeypatch.setattr(synthetic, "generate_pair", lambda *a: (np.ones(400), np.ones(400), metadata))
    monkeypatch.setattr(synthetic, "LENGTHS", (10,))
    def extract(*args, **kwargs):
        return {"starts": np.array([0, 100]), "candidate_starts": np.array([0, 100]),
                "feature_names": ["a", "b"], "values": np.ones((2, 2)),
                "aggregates": np.ones((4, 2))}
    monkeypatch.setitem(sys.modules, "scripts.window_duration_pilot", SimpleNamespace(extract_grid=extract))
    result = synthetic.run_trace((10, "sine_1", 0.1, 0, tmp_path, "frozen"))
    assert result["reused"] is False
    assert unrelated.read_bytes() == b"keep me"
    assert not (tmp_path / f"{stem}.csv.partial").exists()
    assert not (tmp_path / f"{stem}.json.partial").exists()
    assert synthetic.verified_trace(tmp_path, stem, "frozen")["reused"]
