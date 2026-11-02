"""Read-only resource inventory and conservative CPU budgets for the CPU pipeline."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path


def _read_number(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def available_cpu_count() -> int:
    """Respect affinity and common cgroup quotas instead of host logical CPUs."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            available = min(available, max(1, math.floor(int(quota) / int(period))))
    except (OSError, ValueError, ZeroDivisionError):
        quota = _read_number(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
        period = _read_number(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
        if quota is not None and quota > 0 and period and period > 0:
            available = min(available, max(1, quota // period))
    return max(1, available)


def resource_snapshot(volume: Path = Path("/media/volume/catch22")) -> dict:
    """Inspect limits and the actual target filesystem without creating outputs."""
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    limit = _read_number(Path("/sys/fs/cgroup/memory.max"))
    used = _read_number(Path("/sys/fs/cgroup/memory.current"))
    if limit is None:
        limit = _read_number(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        used = _read_number(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    available = memory.get("MemAvailable")
    if limit is not None and used is not None:
        available = min(available, max(0, limit - used)) if available is not None else max(0, limit - used)
    target = volume.expanduser().resolve()
    existing = target
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    disk = shutil.disk_usage(existing)
    return {
        "available_cpus": available_cpu_count(),
        "host_logical_cpus": os.cpu_count(),
        "memory_total_bytes": memory.get("MemTotal"),
        "memory_available_bytes": available,
        "volume": str(target),
        "filesystem_probe": str(existing),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, default=Path("/media/volume/catch22"))
    args = parser.parse_args()
    print(json.dumps(resource_snapshot(args.volume), indent=2))


if __name__ == "__main__":
    main()
