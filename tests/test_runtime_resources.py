from pathlib import Path

from runtime_resources import available_cpu_count, resource_snapshot


def test_cpu_budget_respects_affinity_and_quota(monkeypatch):
    monkeypatch.setattr("runtime_resources.os.sched_getaffinity", lambda _: set(range(8)))
    real_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *a, **kw: "250000 100000" if str(path) ==
                        "/sys/fs/cgroup/cpu.max" else real_read(path, *a, **kw))
    assert available_cpu_count() == 2


def test_resource_inventory_is_read_only_for_missing_destination(tmp_path):
    destination = tmp_path / "future-study"
    snapshot = resource_snapshot(destination)
    assert snapshot["available_cpus"] >= 1
    assert snapshot["disk_free_bytes"] > 0
    assert snapshot["filesystem_probe"] == str(tmp_path)
    assert not destination.exists()
