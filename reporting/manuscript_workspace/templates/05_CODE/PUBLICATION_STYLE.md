# Publication appearance

Edit **05_CODE/publication_style.json** to change the appearance of this offline workspace. All current figure generators and Word table/document writers read it through `publication_style.py`.

The rebuild commands below apply to the complete workspace; the smaller reading packet includes the profile for reference.

The live repository uses the same profile at `reporting/publication_style.json`. At export, the two profile files and helpers are identical. The exported copy is self-contained; it does not need the source VM.

## Rebuild once

From the workspace root, using the bundled Conda writing environment:

```bash
python 05_CODE/rebuild_publication_workspace.py
```

To also write both ZIP files:

```bash
python 05_CODE/rebuild_publication_workspace.py --package
```

The command refreshes all 21 current/reference figures, all native tables, the editable manuscript books, the story map and PDF proofs. It then validates the profile fingerprints, fonts, dimensions, raster quality and Table 2 bolding. It reads bundled data and performs no model fitting, calibration fitting or bootstrap resampling.

## Shared visual meanings

| Variable | Encoding |
| --- | --- |
| Clinical-only input | Blue, solid line |
| Waveform-only input | Orange, dotted line |
| Clinical plus waveform input | Green, dashed line |
| Combined-minus-clinical contrast | Combined-model green |
| Waveform-only-minus-clinical contrast | Waveform orange |
| Whole-case versus windowed in paired channel plots | Filled circle versus open square |
| Outer fold | Shared five-color, line-style and marker sequence, with a fold legend |
| Positive/negative local contribution | Filled/open neutral bars |
| Final combined local prediction | Combined-model green |
| Cohort flow | Monochrome |

Colors are interpreted through the variable named in the legend. A fold-colored curve uses the fold encoding; it is not an input-set comparison. Reference/zero lines, grids, support histograms and table bands have their own neutral tokens. Interval fill and signed-contribution roles are also explicit profile entries.

## Typography and output

- Graphics use DejaVu Sans for portable rendering. The native figure width is 174 mm; most labels are 8–9 pt, with larger cohort-box text where appropriate.
- Word uses Arial, dark headings and 9 pt native table cells. Explicit formatting removes Word theme overrides; bold numerical ranks remain intact.
- Excel keeps its larger Arial sizes for screen editing.
- SVG/PDF/EPS remain vector; PNG/TIFF use 600 dpi and JPG uses 300 dpi. TIFF is lossless RGB; JPG is an editing/review preview.
- Per-figure geometry, axis ranges, units, density and uncertainty displays remain figure-specific. Those carry scientific meaning and should not be flattened into one layout.

## Adding a figure

Use `style_context()` or `apply_mpl_style()`, and use `INPUT_COLORS`, `INPUT_STYLES`, `FOLD_COLORS`, `FOLD_STYLES` and `color()` for semantic roles. Curve, grid and reference widths have shared tokens. Use `apply_document_style()` for editable Word output.

The profile is the appearance source; result CSVs and Methods remain the scientific sources. A style update must preserve all plotted values, input ordering, comparisons and interval definitions. Substantial font/size changes still require a layout check.

## Scope and evidence

All current generated figures in `03_FIGURES` are covered, including six reference views outside the 15-figure publication set. Frozen historical source snapshots, original imported documents and earlier ZIP versions retain their original appearance for provenance.

Every figure manifest records `style_profile_sha256`. Current checks are in `00_START_HERE/qa/publication_style_validation.json`; an isolated override replay demonstrates that changing the profile changed actual SVG colors and Word fonts. `05_CODE/validate_publication_style.py` detects stale figure exports and checks actual Word styles and Table 2 bolding.

## Reader language

Titles name the content. Labels give the quantity and units. Captions define the comparison and visual keys; table notes define units, denominators, intervals and abbreviations. Shared methods belong in Methods; provenance and audit details belong in metadata. Avoid repeated scope adjectives, instructions to the reader and generic methodological disclaimers. Retain qualifications that change interpretation.

Figure manifests and caption files supply the books and figure aliases. Workbook titles and notes come from the tables’ Markdown, so wording stays synchronized. `catalogue_reader_text.py` collects all current/reference SVG text and native table headings/keys in `00_START_HERE/qa/reader_text_inventory.json` for review. Historical source records and machine column names are preserved.

## Navigation and supporting data

Long table PDFs retain their A4 geometry and expose section links and bookmarks. Word books and tables use matching bookmarks and internal links. Figure 3 adds held-out predicted-risk distributions; its count CSV accompanies the existing calibration-bin CSV. Tables S8 and S9 provide calibration diagnostics and the waveform descriptor key.

Display rebuilds preserve existing author manuscript files. Use the explicit author-material refresh option only when replacing the generated starting draft is intended.
