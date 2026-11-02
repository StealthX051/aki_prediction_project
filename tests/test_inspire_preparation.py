"""Preparation checks use synthetic release files only."""
import gzip
import hashlib
import json

import pytest

from scripts.prepare_inspire import RELEASE_FILES, prepare_release


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    # These byte strings deliberately retain negative IDs, quoting and CRLF.
    operations = b'case_id,subject_id,note\r\n-32763,100000001,"a,b"\r\n32767,100000002,"two\nlines"\r\n'
    for name in sorted(RELEASE_FILES):
        data = operations if name == "operations.csv.gz" else b"name,value\r\nx,1\r\n"
        if name.endswith(".gz"):
            data = gzip.compress(data)
        (root / name).write_bytes(data)
    checksums = [f"{hashlib.sha256((root / name).read_bytes()).hexdigest()} {name}\n"
                 for name in sorted(RELEASE_FILES)]
    (root / "SHA256SUMS.txt").write_text("".join(checksums))
    return root, operations


def test_bad_checksum_rejected_before_output_mutation(release, tmp_path):
    root, _ = release
    # A late manifest entry must still fail before any table is extracted.
    (root / "ward_vitals.csv.gz").write_bytes(b"incomplete download")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="checksum mismatch: ward_vitals"):
        prepare_release(root, output)
    assert not output.exists()


def test_roundtrip_preserves_originals_signed_ids_and_csv_records(release, tmp_path):
    root, operations = release
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    output = tmp_path / "prepared"
    prepare_release(root, output)
    assert (output / "operations.csv").read_bytes() == operations
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    manifest = json.loads((output / "preparation_manifest.json").read_text())
    table = manifest["extracted"]["operations.csv"]
    assert table["rows"] == 2  # Embedded newline is part of one quoted field.
    assert table["header"] == ["case_id", "subject_id", "note"]
    assert table["sha256"] == hashlib.sha256(operations).hexdigest()
    assert table["bytes"] == len(operations)
    assert manifest["all_release_checksums_verified"]
    with pytest.raises(ValueError, match="fresh"):
        prepare_release(root, output)
