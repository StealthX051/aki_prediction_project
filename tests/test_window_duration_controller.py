"""Integrity checks use only small, fictional checkpoint fixtures."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import window_duration_controller as controller
from scripts import window_duration_pilot as pilot


@pytest.fixture
def checkpoint_root(tmp_path):
    channels = list(pilot.CHANNELS[:2])
    manifest = {
        "n_patients": 2, "channels": channels, "lengths": [10],
        "feature_names": [f"feature_{index}" for index in range(24)],
        "samples": [{"index": index, "patient_id": 100 + index, "case_id": 200 + index}
                    for index in range(2)],
    }
    shape = (2, 2, 1, 4, 4, 24)
    rng = np.random.default_rng(91)
    arrays = {field: rng.standard_normal(shape) for field in
              ("aggregates", "audited_aggregates", "matched")}
    arrays["phases"] = rng.standard_normal((3, *shape))
    for array in arrays.values():
        array.flat[0] = np.nan
    (tmp_path / "channels").mkdir()
    for sample in manifest["samples"]:
        index = sample["index"]
        for ci, channel in enumerate(channels):
            path = tmp_path / "channels" / f"{index:03d}__{ci}.npz"
            np.savez_compressed(path, **{
                field: array[:, index, ci] if field == "phases" else array[index, ci]
                for field, array in arrays.items()})
            path.with_suffix(".json").write_text(json.dumps({
                "sample_index": index, "channel": channel, "output_sha256": pilot.sha(path)}))
    (tmp_path / "control").mkdir()
    (tmp_path / "control/manifest.json").write_text(json.dumps(manifest))
    np.savez_compressed(tmp_path / "pilot.npz", **arrays,
                        lengths=np.asarray(manifest["lengths"]), channels=np.asarray(channels),
                        feature_names=np.asarray(manifest["feature_names"]),
                        patient_ids=np.array([100, 101]), case_ids=np.array([200, 201]))
    return tmp_path, manifest, arrays


def test_valid_assembly_is_read_only_and_loads_each_npz_once(checkpoint_root, monkeypatch):
    root, manifest, expected = checkpoint_root
    original = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    loads = []
    actual_load = np.load

    def tracked_load(path, *args, **kwargs):
        loads.append(Path(path))
        return actual_load(path, *args, **kwargs)

    monkeypatch.setattr(controller.np, "load", tracked_load)
    checkpoints, actual = controller.verify_pilot_assembly(root)
    assert len(checkpoints) == 4
    assert len(loads) == len(set(loads)) == 5
    for field in expected:
        np.testing.assert_array_equal(actual[field], expected[field])
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == original


@pytest.mark.parametrize("other", ["001__0", "000__1"])
def test_intact_swapped_checkpoint_pairs_fail_identity(checkpoint_root, other):
    root, manifest, _ = checkpoint_root
    first = root / "channels/000__0.npz"
    second = root / "channels" / f"{other}.npz"
    for suffix in (".npz", ".json"):
        left, right = first.with_suffix(suffix), second.with_suffix(suffix)
        saved = left.read_bytes()
        left.write_bytes(right.read_bytes())
        right.write_bytes(saved)
    # Every NPZ still matches the checksum in its paired metadata.
    assert pilot.sha(first) == json.loads(first.with_suffix(".json").read_text())["output_sha256"]
    with pytest.raises(ValueError, match="identity mismatch"):
        controller.verify_pilot_assembly(root, manifest)
    with pytest.raises(ValueError, match="identity mismatch"):
        pilot.process_channel((root, {"index": 0}, manifest["channels"][0]))


@pytest.mark.parametrize("field", ["aggregates", "audited_aggregates", "matched", "phases"])
def test_changed_assembled_payload_fails_before_summary(checkpoint_root, field):
    root, manifest, _ = checkpoint_root
    with np.load(root / "pilot.npz", allow_pickle=False) as bundle:
        changed = {name: bundle[name] for name in bundle.files}
    changed[field].flat[-1] += 1
    np.savez_compressed(root / "pilot.npz", **changed)
    with pytest.raises(ValueError, match=f"assembly differs.*{field}"):
        controller.verify_pilot_assembly(root, manifest)
    assert not (root / "summary").exists()


def test_changed_channel_payload_fails_checksum(checkpoint_root):
    root, manifest, _ = checkpoint_root
    path = root / "channels/000__0.npz"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        controller.verify_pilot_assembly(root, manifest)
