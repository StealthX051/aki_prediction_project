"""Verify and extract an INSPIRE release without modifying its original files.

No identifiers, values, labels or rows are transformed. A fresh output directory
contains decompressed CSVs and an aggregate provenance manifest.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re


RELEASE_FILES = {
    "CHANGELOG.txt", "LICENSE.txt", "department.csv", "diagnosis.csv.gz",
    "icd10_excluded.csv", "labs.csv.gz", "medications.csv.gz",
    "operations.csv.gz", "parameters.csv", "schema.csv", "vitals.csv.gz",
    "ward_vitals.csv.gz",
}
BLOCK_BYTES = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_metadata(path: Path) -> dict:
    """Count CSV records without coercing identifiers or measured values."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = csv.reader(stream, strict=True)
        header = next(records, [])
        count = sum(1 for row in records if row)
    return {"header": header, "rows": count}


def prepare_release(release_dir: Path, output_dir: Path) -> dict:
    release_dir = release_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Output directory must be fresh and absent")
    if output_dir.is_relative_to(release_dir):
        raise ValueError("Output directory must be outside the original release")
    checksum_path = release_dir / "SHA256SUMS.txt"
    checksum_digest = sha256(checksum_path)
    expected = {}
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name in expected or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("Invalid or duplicate release checksum entry")
        expected[name] = digest.lower()
    if set(expected) != RELEASE_FILES:
        raise ValueError("Release manifest must cover all 12 expected INSPIRE files")

    originals = {}
    # Verify the entire release before creating any output, even for small tables.
    for name, digest in expected.items():
        path = release_dir / name
        observed = sha256(path)
        if observed != digest:
            raise ValueError(f"Release checksum mismatch: {name}")
        originals[name] = {"sha256": observed, "bytes": path.stat().st_size}
    if sha256(checksum_path) != checksum_digest:
        raise RuntimeError("Release checksum manifest changed during verification")

    output_dir.mkdir(parents=True, exist_ok=False)
    extracted = {}
    for name in expected:
        source = release_dir / name
        if not name.endswith(".csv.gz"):
            if name.endswith(".csv"):
                originals[name].update(csv_metadata(source))
            continue
        destination = output_dir / name.removesuffix(".gz")
        partial = destination.with_suffix(destination.suffix + ".partial")
        digest = hashlib.sha256()
        size = 0
        # Reading to EOF checks the gzip CRC and uncompressed length. Preserve
        # original bytes, including signed identifiers, quoting and line endings.
        with gzip.open(source, "rb") as reader, partial.open("xb") as writer:
            for block in iter(lambda: reader.read(BLOCK_BYTES), b""):
                writer.write(block)
                digest.update(block)
                size += len(block)
        metadata = csv_metadata(partial)
        partial.rename(destination)
        extracted[destination.name] = {
            "source": name, "sha256": digest.hexdigest(), "bytes": size,
            "gzip_integrity_verified": True, **metadata,
        }
        originals[name].update(metadata)

    # Changes during preparation invalidate provenance; no complete manifest is
    # published in that event. Original files are never opened for writing.
    if sha256(checksum_path) != checksum_digest or any(
        sha256(release_dir / name) != entry["sha256"]
        for name, entry in originals.items()
    ):
        raise RuntimeError("An original release file changed during preparation")
    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(release_dir), "output_dir": str(output_dir),
        "release_checksum_manifest_sha256": checksum_digest,
        "all_release_checksums_verified": True,
        "preparation": "Byte-preserving gzip extraction only; no identifier, value, label or row transformations.",
        "originals": originals, "extracted": extracted,
    }
    partial_manifest = output_dir / "preparation_manifest.json.partial"
    with partial_manifest.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    partial_manifest.rename(output_dir / "preparation_manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare_release(args.release_dir, args.output_dir)
    print(f"Verified {len(manifest['originals'])} originals; extracted "
          f"{len(manifest['extracted'])} tables to {args.output_dir}")


if __name__ == "__main__":
    main()
