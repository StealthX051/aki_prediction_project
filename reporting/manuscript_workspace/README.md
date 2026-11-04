# Manuscript workspace code

This directory versions the code used by the complete manuscript-workspace ZIP.
`templates/` mirrors its code locations; the scripts run after being copied into
that workspace. Research data, generated displays, imported documents and ZIPs
remain outside Git.

From the repository root, compare an unpacked complete workspace with this code:

```bash
python -m reporting.manuscript_workspace /path/to/manuscript-workspace --check
```

To update its code and writing configuration, then rebuild its displays:

```bash
python -m reporting.manuscript_workspace /path/to/manuscript-workspace --apply
cd /path/to/manuscript-workspace
python 05_CODE/rebuild_publication_workspace.py --package
```

The check is read-only. `--apply` replaces only the versioned code/configuration
paths; it preserves data, generated displays and author files. A rebuild preserves
existing manuscript starting documents by default. Running
`python 05_CODE/build_manuscript_documents.py --refresh-author-material`
explicitly regenerates those starting files.
Run the display rebuild in the bundled writing Conda environment; PDF tools
Ghostscript and Poppler are also required.

The shared appearance files come from `reporting/publication_style.py` and
`reporting/publication_style.json`, so this template set carries no second copy.
Other templates cover the figures, native tables, document books, navigation,
source checks and packaging. Edit templates here for future code changes; use the
workspace copies to rebuild the paper.

`05_CODE/export_heldout_cases.py` is optional: selecting new cases requires trusted
saved models and the research repository. The normal rebuild uses bundled case
CSVs and performs no model, calibrator or threshold fitting.
