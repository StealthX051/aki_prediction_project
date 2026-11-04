# Shared publication style

`publication_style.json` is the appearance source for current reporting helpers. It defines the input, family and fold palettes, branch markers, neutral roles, graphics/Office fonts, line weights and export quality. The JSON is versioned in Git.

Use `report_figure_style_context()` for existing large report layouts. For a native publication figure, use `publication_style.style_context()` and the shared input/fold encodings. Keep geometry, data transformations, axes and uncertainty definitions local to the scientific figure.

Word writers use `apply_document_style()` to resolve Arial and dark headings explicitly, including removal of template theme overrides. Numerical bolding is retained. Excel retains its larger editing font sizes; graphics use portable DejaVu Sans.

The offline manuscript workspace carries identical `publication_style.py` and `.json` files in `05_CODE`. Its figure and document writers use that local copy. `05_CODE/rebuild_publication_workspace.py` rebuilds the complete current display set and validates appearance fingerprints, fonts, dimensions, raster quality and Table 2 bolding. Add `--package` to refresh the reading and full ZIPs.

Historical snapshots and imported documents retain their original formatting. Review layouts after substantial font or figure-size changes.

The [manuscript workspace templates](manuscript_workspace/README.md) are versioned here. Use their installer to update an unpacked writing workspace with the current code and this shared profile.
