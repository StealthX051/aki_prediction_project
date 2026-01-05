"""Create a cohort construction flow diagram from saved counts.

The flow diagram visualizes how many cases remain after each filtering step in
``data_preparation.step_01_cohort_construction``. The module accepts structured
counts saved to JSON (for example by persisting print statements from
``step_01``) and renders a simple vertical flow chart to ``results/figures``.

Expected JSON structure (flexible):

```
{
    "total_cases": 6388,
    "departments": {"count": 5320, "departments": ["general surgery"]},
    "mandatory_columns": {"count": 4890, "columns": ["preop_cr", "opend"]},
    "waveforms": [
        {"channel": "SNUADC/PLETH", "count": 4720},
        {"channel": "SNUADC/ECG_II", "count": 4680}
    ],
    "custom_filters": [
        {"name": "filter_preop_cr", "label": "Baseline Cr available", "count": 4510}
    ],
    "final_cohort": {"count": 4510}
}
```

Only ``total_cases`` is strictly required; other keys are optional but will be
rendered in the order shown above. Waveform labels fall back to
``DisplayDictionary.waveform_label`` when available.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple, Union
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from reporting.display_dictionary import DisplayDictionary, load_display_dictionary

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))
FIGURES_DIR = RESULTS_DIR / "figures"
METADATA_DIR = RESULTS_DIR / "metadata"
DEFAULT_COUNTS_PATH = METADATA_DIR / "cohort_flow_counts.json"


@dataclass(frozen=True)
class CohortStage:
    """Normalized representation of a cohort filtering stage."""

    title: str
    count: int
    removed: Optional[int] = None


@dataclass(frozen=True)
class OutcomeSplit:
    """Optional terminal split (e.g., AKI label false/true)."""

    label: str
    true_count: int
    false_count: int

    @property
    def total(self) -> int:
        return self.true_count + self.false_count


@dataclass(frozen=True)
class CohortFlowData:
    """Container holding ordered stages and an optional outcome split."""

    stages: List[CohortStage]
    outcome_split: Optional[OutcomeSplit] = None


def _load_counts(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cohort counts file not found at {path}. Provide --counts-file to override."
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_count(entry: Any) -> Optional[int]:
    if isinstance(entry, Mapping):
        value = entry.get("count")
    else:
        value = entry

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("Could not parse count value %s", value)
        return None


def _format_detail(prefix: str, values: Iterable[str]) -> str:
    joined = ", ".join(values)
    return f"{prefix}: {joined}" if joined else prefix


def _waveform_label(channel: str, display_dictionary: Optional[DisplayDictionary]) -> str:
    if display_dictionary:
        try:
            return display_dictionary.waveform_label(channel, use_short=True)
        except Exception:  # DisplayDictionary has no strict validation
            logger.debug("Falling back to raw waveform key for %s", channel)
    return channel


def normalize_counts(
    raw_counts: Mapping[str, Any], display_dictionary: Optional[DisplayDictionary] = None
) -> CohortFlowData:
    """Convert raw count metadata into ordered :class:`CohortStage` entries."""

    filter_label_map = {
        "filter_preop_cr": "Preop creatinine available",
        "ensure_sample_independence": "Sampling independence",
        "filter_postop_cr": "Postop creatinine available",
        "add_aki_label": "AKI label derived",
    }

    def _parse_outcome_split(raw_counts_inner: Mapping[str, Any]) -> Optional[OutcomeSplit]:
        split_raw = (
            raw_counts_inner.get("outcome_split")
            or raw_counts_inner.get("label_split")
            or raw_counts_inner.get("label_counts")
        )
        if not split_raw or not isinstance(split_raw, Mapping):
            return None

        true_count = _extract_count(split_raw.get("true") or split_raw.get("positive"))
        false_count = _extract_count(split_raw.get("false") or split_raw.get("negative"))
        if true_count is None or false_count is None:
            return None

        label = str(split_raw.get("label") or "Label split")
        return OutcomeSplit(label=label, true_count=true_count, false_count=false_count)

    stages: MutableSequence[CohortStage] = []

    total_cases = _extract_count(raw_counts.get("total_cases") or raw_counts.get("total"))
    if total_cases is None:
        raise ValueError("Counts must include a 'total_cases' entry.")
    stages.append(CohortStage(title="Total cases", count=total_cases))

    departments = raw_counts.get("departments")
    dept_count = _extract_count(departments)
    if dept_count is not None:
        dept_values = []
        if isinstance(departments, Mapping):
            dept_values = [str(d) for d in departments.get("departments", [])]
        title = "Department filter"
        if dept_values:
            title = _format_detail(title, dept_values)
        stages.append(CohortStage(title=title, count=dept_count))

    mandatory = raw_counts.get("mandatory_columns")
    mandatory_count = _extract_count(mandatory)
    if mandatory_count is not None:
        columns = []
        if isinstance(mandatory, Mapping):
            columns = [str(col) for col in mandatory.get("columns", [])]
        title = "Mandatory columns present"
        if columns:
            title = _format_detail(title, columns)
        stages.append(CohortStage(title=title, count=mandatory_count))

    waveforms = raw_counts.get("waveforms") or []
    if isinstance(waveforms, Mapping):
        waveforms = [
            {"channel": key, "count": value} for key, value in waveforms.items()
        ]

    for entry in waveforms:
        if isinstance(entry, Mapping):
            channel = str(entry.get("channel") or entry.get("waveform") or entry.get("track") or "Waveform")
            count = _extract_count(entry)
        else:
            logger.warning("Skipping unrecognized waveform entry: %s", entry)
            continue

        if count is None:
            continue

        label = _waveform_label(channel, display_dictionary)
        stages.append(CohortStage(title=f"Waveform availability — {label}", count=count))

    custom_filters = raw_counts.get("custom_filters") or []
    if isinstance(custom_filters, Mapping):
        custom_filters = [
            {"name": key, "count": value} for key, value in custom_filters.items()
        ]

    for entry in custom_filters:
        if isinstance(entry, Mapping):
            raw_name = str(entry.get("name") or "Custom filter")
            label = str(entry.get("label") or filter_label_map.get(raw_name) or raw_name.replace("_", " ").title())
            name = label
            count = _extract_count(entry)
        else:
            logger.warning("Skipping unrecognized custom filter entry: %s", entry)
            continue

        if count is None:
            continue
        stages.append(CohortStage(title=name, count=count))

    final_cohort = raw_counts.get("final_cohort") or raw_counts.get("final")
    final_count = _extract_count(final_cohort)
    if final_count is None and stages:
        final_count = stages[-1].count

    if final_count is not None:
        stages.append(CohortStage(title="Final cohort", count=final_count))

    outcome_split = _parse_outcome_split(raw_counts)

    # Drop non-filtering stages (no change) except first and final; attach removed deltas.
    filtered: List[CohortStage] = []
    prev_count: Optional[int] = None
    for idx, stage in enumerate(stages):
        if prev_count is None:
            filtered.append(stage)
            prev_count = stage.count
            continue

        delta = prev_count - stage.count
        is_final = stage.title == "Final cohort" or idx == len(stages) - 1
        if delta == 0 and not is_final:
            # Skip no-op steps
            continue
        if delta < 0 and not is_final:
            logger.warning("Skipping stage %s because count increased (%s -> %s)", stage.title, prev_count, stage.count)
            continue

        filtered.append(CohortStage(title=stage.title, count=stage.count, removed=delta if delta > 0 else None))
        prev_count = stage.count

    return CohortFlowData(stages=filtered, outcome_split=outcome_split)


def render_cohort_flow(
    flow: Union[Sequence[CohortStage], CohortFlowData],
    *,
    output_dir: Path = FIGURES_DIR,
    output_name: str = "cohort_flow",
    title: Optional[str] = "Cohort Construction Flow",
    formats: Sequence[str] = ("svg", "png"),
) -> List[Path]:
    """Render the cohort flow chart to the requested formats."""

    if isinstance(flow, CohortFlowData):
        stages = flow.stages
        outcome_split = flow.outcome_split
    else:
        stages = list(flow)
        outcome_split = None

    if not stages:
        raise ValueError("At least one stage is required to render a flow diagram.")

    output_dir.mkdir(parents=True, exist_ok=True)

    vertical_steps = len(stages) + (1 if outcome_split else 0)
    fig_height = max(6.0, (1.8 * vertical_steps))
    fig_width = 8.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    box_width = 6.4
    box_height = 1.0
    gap = 0.8
    x0 = 1.0
    y_top = (box_height + gap) * (len(stages) - 1) + 0.5
    lowest_y = y_top

    for idx, stage in enumerate(stages):
        y = y_top - idx * (box_height + gap)
        lowest_y = y
        box = FancyBboxPatch(
            (x0, y),
            box_width,
            box_height,
            boxstyle="round,pad=0.35,rounding_size=0.15",
            linewidth=1.2,
            facecolor="#f8fafc",
            edgecolor="#1f2937",
        )
        ax.add_patch(box)

        wrapped_title = "\n".join(textwrap.wrap(stage.title, width=32)) or stage.title
        removed_text = f"\nRemoved: {stage.removed:,}" if stage.removed else ""
        text = f"{wrapped_title}\n n = {stage.count:,}{removed_text}"
        ax.text(
            x0 + box_width / 2,
            y + box_height / 2,
            text,
            ha="center",
            va="center",
            fontsize=11.5,
            wrap=True,
        )

        if idx < len(stages) - 1:
            arrow = FancyArrowPatch(
                (x0 + box_width / 2, y - 0.05),
                (x0 + box_width / 2, y - gap + 0.05),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color="#64748b",
                shrinkA=4,
                shrinkB=4,
            )
            ax.add_patch(arrow)

        split_box_width = 4.0
        split_spacing = 0.75

    if outcome_split:
        split_y = lowest_y - (box_height + gap + 0.2)
        total = outcome_split.total or 1

        left_x = x0 - split_box_width - split_spacing
        right_x = x0 + box_width + split_spacing

        def _draw_split_box(x: float, label: str, count: int) -> None:
            pct = (count / total) * 100
            box = FancyBboxPatch(
                (x, split_y),
                split_box_width,
                box_height,
                boxstyle="round,pad=0.35",
                linewidth=1.5,
                facecolor="#f8fafc",
                edgecolor="#0f172a",
            )
            ax.add_patch(box)
            ax.text(
                x + split_box_width / 2,
                split_y + box_height / 2,
                f"{label}\n{count:,} ({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=10.5,
            )

        _draw_split_box(left_x, f"{outcome_split.label} — False", outcome_split.false_count)
        _draw_split_box(right_x, f"{outcome_split.label} — True", outcome_split.true_count)

        # Arrows from final cohort to split boxes
        final_center_x = x0 + box_width / 2
        final_center_y = lowest_y
        for target_x in (left_x + split_box_width / 2, right_x + split_box_width / 2):
            arrow = FancyArrowPatch(
                (final_center_x, final_center_y - 0.05),
                (target_x, split_y + box_height + 0.05),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#475569",
            )
            ax.add_patch(arrow)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    max_x = x0 + box_width + split_box_width + 2 * split_spacing if outcome_split else x0 + box_width + 1
    min_x = (x0 - split_box_width - split_spacing) if outcome_split else 0
    ax.set_xlim(min_x - 0.5, max_x + 0.5)
    ax.set_ylim(-gap - box_height, y_top + box_height + gap)

    saved_paths: List[Path] = []
    for fmt in formats:
        path = output_dir / f"{output_name}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        saved_paths.append(path)
        logger.info("Saved cohort flow diagram to %s", path)

    plt.close(fig)
    return saved_paths


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts-file",
        type=Path,
        default=DEFAULT_COUNTS_PATH,
        help="Path to JSON file containing cohort counts.",
    )
    parser.add_argument(
        "--display-dictionary",
        type=Path,
        default=None,
        help="Optional override path for metadata/display_dictionary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory to write output figures.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="cohort_flow",
        help="Base filename (without extension) for the diagram.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Cohort Construction Flow",
        help="Optional title for the figure.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("svg", "png"),
        help="Figure formats to render (e.g., svg png).",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = _build_argument_parser()
    args = parser.parse_args()

    try:
        display_dict = load_display_dictionary(args.display_dictionary) if args.display_dictionary else load_display_dictionary()
    except FileNotFoundError:
        logger.warning("Display dictionary not found; waveform labels will use raw keys.")
        display_dict = None

    counts = _load_counts(args.counts_file)
    flow_data = normalize_counts(counts, display_dict)
    render_cohort_flow(
        flow_data,
        output_dir=args.output_dir,
        output_name=args.output_name,
        title=args.title,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
