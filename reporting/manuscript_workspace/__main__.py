"""Check or update code in an existing exported manuscript workspace."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
REPORTING = HERE.parent
MARKERS = ("00_START_HERE/READ_ME_FIRST.txt", "04_DATA/prediction_index.csv")


def source_files() -> dict[Path, Path]:
    """Map fixed workspace destinations to versioned source; include shared style once."""
    files = {
        path.relative_to(TEMPLATES): path
        for path in sorted(TEMPLATES.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    for name in ("publication_style.py", "publication_style.json"):
        files[Path("05_CODE") / name] = REPORTING / name
    return files


def synchronize(workspace: Path, *, apply: bool = False) -> list[Path]:
    """Return differing code paths; replace only those paths when explicitly requested."""
    workspace = workspace.resolve()
    if not all((workspace / marker).is_file() for marker in MARKERS):
        raise ValueError("Expected an unpacked complete manuscript workspace (not the reading packet).")
    sources = source_files()
    # Validate every destination before writing, including parent-directory symlinks.
    for relative in sources:
        target = workspace / relative
        if target.resolve() != target:
            raise ValueError(f"Code destination uses a symbolic link: {relative}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Code destination is not a file: {relative}")
    differences = [
        relative for relative, source in sources.items()
        if not (workspace / relative).is_file()
        or (workspace / relative).read_bytes() != source.read_bytes()
    ]
    if apply:
        for relative in differences:
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sources[relative], target)
    return differences


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="Root of an unpacked complete-workspace ZIP")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only comparison (the default)")
    mode.add_argument("--apply", action="store_true", help="Update code/configuration files only")
    args = parser.parse_args(argv)
    try:
        differences = synchronize(args.workspace, apply=args.apply)
    except ValueError as error:
        parser.error(str(error))
    verb = "Updated" if args.apply else "Different"
    print(f"{len(source_files())} source files checked; {len(differences)} {verb.lower()}.")
    for relative in differences:
        print(f"{verb}: {relative}")
    if args.apply:
        print("Data, figures, tables and author manuscripts were preserved. Rebuild separately when needed.")
    return int(bool(differences) and not args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
