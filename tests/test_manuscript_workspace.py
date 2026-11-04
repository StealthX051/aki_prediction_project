"""The code-only handoff must preserve the author's data and writing."""
from pathlib import Path

import pytest

from reporting.manuscript_workspace.__main__ import MARKERS, main, source_files, synchronize


def make_workspace(root):
    for marker in MARKERS:
        path = root / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("workspace marker\n")
    return root


def test_check_is_read_only_and_apply_preserves_author_data(tmp_path):
    workspace = make_workspace(tmp_path / "workspace")
    author = workspace / "01_MANUSCRIPT/author_material/Manuscript_Starting_Document.docx"
    author.parent.mkdir(parents=True)
    author.write_bytes(b"author edits")
    data = workspace / "04_DATA/predictions.csv"
    data.write_bytes(b"caseid,y_prob\n1,0.4\n")
    old_code = workspace / "05_CODE/rebuild_publication_workspace.py"
    old_code.parent.mkdir()
    old_code.write_bytes(b"old code")
    before = {p.relative_to(workspace): p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    assert main([str(workspace), "--check"]) == 1
    assert before == {p.relative_to(workspace): p.read_bytes() for p in workspace.rglob("*") if p.is_file()}
    assert main([str(workspace), "--apply"]) == 0
    assert main([str(workspace), "--check"]) == 0
    for relative, content in before.items():
        if relative != old_code.relative_to(workspace):
            assert (workspace / relative).read_bytes() == content
    assert old_code.read_bytes() == source_files()[old_code.relative_to(workspace)].read_bytes()


def test_refuses_missing_workspace_and_symlink_destination(tmp_path):
    with pytest.raises(ValueError, match="unpacked complete"):
        synchronize(tmp_path, apply=True)
    workspace = make_workspace(tmp_path / "workspace")
    code = workspace / "05_CODE"
    code.mkdir()
    outside = tmp_path / "writing.py"
    outside.write_bytes(b"protected")
    (code / "rebuild_publication_workspace.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        synchronize(workspace, apply=True)
    assert outside.read_bytes() == b"protected"
    assert not (code / "build_manuscript_documents.py").exists()


def test_templates_are_source_only_and_parse():
    files = source_files()
    assert files[Path("05_CODE/publication_style.json")].parent.name == "reporting"
    assert files[Path("05_CODE/publication_style.py")].parent.name == "reporting"
    for relative, source in files.items():
        assert relative.parts[0] in {"02_TABLES", "03_FIGURES", "05_CODE"}
        assert source.suffix in {".py", ".json", ".md", ".txt", ".yml"}
        if source.suffix == ".py":
            compile(source.read_text(), str(relative), "exec")
