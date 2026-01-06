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
from matplotlib.patches import FancyBboxPatch

from reporting.display_dictionary import DisplayDictionary, load_display_dictionary

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(
    os.getenv("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results" / "catch22" / "experiments")
)
PAPER_DIR = Path(os.getenv("PAPER_DIR", EXPERIMENTS_DIR.parent / "paper"))
FIGURES_DIR = PAPER_DIR / "figures"
METADATA_DIR = PAPER_DIR / "metadata"
DEFAULT_COUNTS_PATH = METADATA_DIR / "cohort_flow_counts.json"


@dataclass(frozen=True)
class CohortStage:
    """Normalized representation of a cohort filtering stage."""

    title: str
    count: int
    detail: Optional[str] = None
    removed: Optional[int] = None
    removal_reason: Optional[str] = None


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

    column_label_map = {
        "preop_cr": "Preoperative creatinine",
        "opend": "Documented anesthesia end time",
    }

    filter_label_map = {
        "filter_preop_cr": "Baseline Renal Function Data",
        "ensure_sample_independence": "Subject Selection",
        "filter_postop_cr": "Outcome Data Availability",
        "add_aki_label": "AKI label derived",
    }

    removal_reason_map = {
        "mandatory_columns": "Missing mandatory data fields",
        "waveforms": "Incomplete waveform data",
        "filter_preop_cr": "Missing preoperative creatinine",
        "ensure_sample_independence": "Exclusion of repeat encounters",
        "filter_postop_cr": "Missing postoperative creatinine",
    }

    title_overrides = {
        "mandatory_columns": "EHR Record Screening",
        "waveforms": "High-fidelity Waveform Availability",
        "filter_preop_cr": "Baseline Renal Function Data",
        "ensure_sample_independence": "Subject Selection",
        "filter_postop_cr": "Outcome Data Availability",
    }

    def _pretty_column(name: str) -> str:
        return column_label_map.get(name, name.replace("_", " "))

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
        pretty_columns = [_pretty_column(col) for col in columns]
        title = title_overrides.get("mandatory_columns", "Initial data screening")
        detail = None
        if pretty_columns:
            detail = _format_detail("Mandatory data fields", pretty_columns)
        stages.append(
            CohortStage(
                title=title,
                detail=detail,
                count=mandatory_count,
                removal_reason=removal_reason_map.get("mandatory_columns"),
            )
        )

    waveforms = raw_counts.get("waveforms") or []
    if isinstance(waveforms, Mapping):
        waveforms = [
            {"channel": key, "count": value} for key, value in waveforms.items()
        ]

    waveform_entries: List[Tuple[str, int]] = []
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
        waveform_entries.append((label, count))

    if waveform_entries:
        final_waveform_count = waveform_entries[-1][1]
        labels = [label for label, _ in waveform_entries]
        detail = _format_detail("Required waveforms", labels)
        stages.append(
            CohortStage(
                title=title_overrides.get("waveforms", "Waveform availability"),
                count=final_waveform_count,
                detail=detail,
                removal_reason=removal_reason_map.get("waveforms"),
            )
        )

    custom_filters = raw_counts.get("custom_filters") or []
    if isinstance(custom_filters, Mapping):
        custom_filters = [
            {"name": key, "count": value} for key, value in custom_filters.items()
        ]

    for entry in custom_filters:
        if isinstance(entry, Mapping):
            raw_name = str(entry.get("name") or "Custom filter")
            label = str(
                entry.get("label")
                or title_overrides.get(raw_name)
                or filter_label_map.get(raw_name)
                or raw_name.replace("_", " ").title()
            )
            name = label
            count = _extract_count(entry)
        else:
            logger.warning("Skipping unrecognized custom filter entry: %s", entry)
            continue

        if count is None:
            continue
        stages.append(
            CohortStage(
                title=name,
                count=count,
                removal_reason=removal_reason_map.get(raw_name),
            )
        )

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

        filtered.append(
            CohortStage(
                title=stage.title,
                count=stage.count,
                detail=stage.detail,
                removal_reason=stage.removal_reason,
                removed=delta if delta > 0 else None,
            )
        )
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

    plt.rcParams.update(
        {
            "font.family": ["DejaVu Sans"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        }
    )

    box_width = 5.8
    box_height = 0.9
    box_gap = 1.05
    extra_gap_after_waveform = 0.25
    box_pad = 0.32
    center_x = 0.0

    removal_box_width = 3.6
    removal_box_height = 0.82
    removal_gap = 0.95  # wider offset for breathing room

    split_box_width = 3.9
    split_box_height = 1.0
    split_offset = box_width * 0.72 + 1.2

    vertical_steps = len(stages) + (1 if outcome_split else 0)
    fig_height = max(8.0, (box_height + box_gap) * (vertical_steps + 1.1))
    fig_width = 12.0
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    def _wrap_line(text: str, width: int = 30) -> str:
        return "\n".join(textwrap.wrap(text, width=width)) if text else ""

    y_positions: List[float] = []
    y_cursor = 0.0
    for stage in stages:
        y_positions.append(y_cursor)
        gap = box_gap + (extra_gap_after_waveform if "waveform" in stage.title.lower() else 0.0)
        y_cursor -= (box_height + gap)

    lowest_y = y_positions[-1]
    min_x = center_x - box_width / 2
    max_x = center_x + box_width / 2

    arrow_zorder = 2
    flow_arrowprops = dict(
        arrowstyle="-|>",
        linewidth=1.5,
        color="#0f172a",
        mutation_scale=20,
        connectionstyle="arc3,rad=0",
        zorder=arrow_zorder,
    )
    exclusion_arrowprops = dict(
        arrowstyle="-|>",
        linewidth=1.4,
        color="#1f2937",
        mutation_scale=20,
        connectionstyle="arc3,rad=0",
        zorder=arrow_zorder,
    )
    split_arrowprops = dict(
        arrowstyle="-|>",
        linewidth=1.6,
        color="#0f172a",
        mutation_scale=20,
        zorder=arrow_zorder,
    )

    def draw_connection(
        ax_obj: plt.Axes,
        start_box: FancyBboxPatch,
        end_box: FancyBboxPatch,
        *,
        renderer: Any,
        direction: str = "down",
        arrow_kwargs: Mapping[str, Any],
        connectionstyle: Optional[str] = None,
    ) -> None:
        """Connect two boxes using their bounds so arrows stay aligned."""

        def _centers(box: FancyBboxPatch) -> Mapping[str, Tuple[float, float]]:
            bbox = box.get_window_extent(renderer=renderer).transformed(ax_obj.transData.inverted())
            x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
            xc = (x0 + x1) / 2
            yc = (y0 + y1) / 2
            return {
                "top": (xc, y1),
                "bottom": (xc, y0),
                "left": (x0, yc),
                "right": (x1, yc),
            }

        start_centers = _centers(start_box)
        end_centers = _centers(end_box)

        if direction == "down":
            start_pt, end_pt = start_centers["bottom"], end_centers["top"]
        elif direction == "up":
            start_pt, end_pt = start_centers["top"], end_centers["bottom"]
        elif direction == "right":
            start_pt, end_pt = start_centers["right"], end_centers["left"]
        elif direction == "left":
            start_pt, end_pt = start_centers["left"], end_centers["right"]
        else:
            start_pt, end_pt = start_centers["bottom"], end_centers["top"]

        arrowprops = dict(arrow_kwargs)
        if connectionstyle:
            arrowprops["connectionstyle"] = connectionstyle

        ax_obj.annotate(
            "",
            xy=end_pt,
            xytext=start_pt,
            arrowprops=arrowprops,
        )

    main_boxes: List[FancyBboxPatch] = []
    connections: List[Tuple[FancyBboxPatch, FancyBboxPatch, str, Mapping[str, Any], Optional[str]]] = []

    for idx, (stage, y_center) in enumerate(zip(stages, y_positions)):
        left = center_x - box_width / 2
        bottom = y_center - box_height / 2
        box = FancyBboxPatch(
            (left, bottom),
            box_width,
            box_height,
            boxstyle=f"round,pad={box_pad},rounding_size=0.12",
            linewidth=1.5,
            facecolor="white",
            edgecolor="black",
            zorder=3,
        )
        ax.add_patch(box)
        main_boxes.append(box)

        lines = [_wrap_line(stage.title, 30)]
        if stage.detail:
            detail_line = _wrap_line(stage.detail, 34)
            if detail_line:
                lines.append(detail_line)
        lines.append(f"n = {stage.count:,}")
        box_text = "\n".join(lines)
        ax.text(
            center_x,
            y_center,
            box_text,
            ha="center",
            va="center",
            fontsize=11.5,
            linespacing=1.3,
            zorder=4,
        )

        if stage.removed:
            removal_left = center_x + box_width / 2 + removal_gap
            removal_bottom = y_center - removal_box_height / 2
            removal_box = FancyBboxPatch(
                (removal_left, removal_bottom),
                removal_box_width,
                removal_box_height,
                boxstyle="round,pad=0.22,rounding_size=0.08",
                linewidth=1.3,
                facecolor="white",
                edgecolor="black",
                zorder=3,
            )
            ax.add_patch(removal_box)

            exclusion_anchor = stage.removal_reason or stage.detail or stage.title
            footnote = None
            if "waveform" in stage.title.lower() and stage.detail:
                footnote = stage.detail.replace("Required waveforms: ", "")

            base_label = f"{exclusion_anchor} (n = {stage.removed:,})"
            removal_lines = [_wrap_line(base_label, 30)]
            if footnote:
                removal_lines.append(_wrap_line(f"*{footnote}", 30))

            ax.text(
                removal_left + removal_box_width / 2,
                y_center,
                "\n".join(removal_lines),
                ha="center",
                va="center",
                fontsize=10.2,
                linespacing=1.22,
                zorder=4,
            )
            max_x = max(max_x, removal_left + removal_box_width)
            connections.append((box, removal_box, "right", exclusion_arrowprops, None))

    for idx in range(len(main_boxes) - 1):
        connections.append((main_boxes[idx], main_boxes[idx + 1], "down", flow_arrowprops, None))

    split_patches: List[FancyBboxPatch] = []

    if outcome_split:
        split_y = lowest_y - (box_height / 2 + box_gap + 0.9)
        total = outcome_split.total or 1

        left_center_x = center_x - split_offset
        right_center_x = center_x + split_offset

        def _split_label(label: str, count: int) -> str:
            pct = (count / total) * 100
            return f"{label}\n{count:,} ({pct:.1f}%)"

        label_lower = outcome_split.label.lower()
        left_label = "No AKI" if "aki" in label_lower else f"{outcome_split.label} — False"
        right_label = "Acute Kidney Injury (AKI)" if "aki" in label_lower else f"{outcome_split.label} — True"

        for box_center_x, label_text, count in (
            (left_center_x, left_label, outcome_split.false_count),
            (right_center_x, right_label, outcome_split.true_count),
        ):
            split_left = box_center_x - split_box_width / 2
            split_bottom = split_y - split_box_height / 2
            split_box = FancyBboxPatch(
                (split_left, split_bottom),
                split_box_width,
                split_box_height,
                boxstyle=f"round,pad={box_pad}",
                linewidth=1.5,
                facecolor="white",
                edgecolor="black",
                zorder=3,
            )
            ax.add_patch(split_box)
            split_patches.append(split_box)
            ax.text(
                box_center_x,
                split_y,
                _split_label(label_text, count),
                ha="center",
                va="center",
                fontsize=11.0,
                linespacing=1.25,
                zorder=4,
            )
            max_x = max(max_x, split_left + split_box_width)
            min_x = min(min_x, split_left)

        final_box = main_boxes[-1]

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _centers_for(box: FancyBboxPatch) -> Mapping[str, Tuple[float, float]]:
        bbox = box.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        xc = (x0 + x1) / 2
        yc = (y0 + y1) / 2
        return {
            "top": (xc, y1),
            "bottom": (xc, y0),
            "left": (x0, yc),
            "right": (x1, yc),
        }

    for start_box, end_box, direction, arrow_kwargs, conn_style in connections:
        if start_box is None or end_box is None:
            continue
        draw_connection(
            ax,
            start_box,
            end_box,
            renderer=renderer,
            direction=direction,
            arrow_kwargs=arrow_kwargs,
            connectionstyle=conn_style,
        )

    if outcome_split and len(split_patches) == 2:
        final_box = main_boxes[-1]
        left_patch, right_patch = split_patches

        final_centers = _centers_for(final_box)
        left_centers = _centers_for(left_patch)
        right_centers = _centers_for(right_patch)

        final_bottom = final_centers["bottom"]
        split_top_y = min(left_centers["top"][1], right_centers["top"][1])
        junction_y = (final_bottom[1] + split_top_y) / 2.0

        # Vertical stem from final cohort
        ax.plot(
            [final_bottom[0], final_bottom[0]],
            [final_bottom[1], junction_y],
            color=flow_arrowprops["color"],
            linewidth=1.6,
            zorder=arrow_zorder,
        )

        # Horizontal bar across split targets
        ax.plot(
            [left_centers["top"][0], right_centers["top"][0]],
            [junction_y, junction_y],
            color=flow_arrowprops["color"],
            linewidth=1.6,
            zorder=arrow_zorder,
        )

        # Downward arrows into split boxes
        for target_centers in (left_centers, right_centers):
            start_pt = (target_centers["top"][0], junction_y)
            end_pt = target_centers["top"]
            ax.annotate(
                "",
                xy=end_pt,
                xytext=start_pt,
                arrowprops=dict(split_arrowprops, connectionstyle="arc3,rad=0"),
            )

    if title:
        ax.set_title(title, fontsize=15, fontweight="bold", pad=14)

    x_padding = 0.8
    y_padding = 0.8
    y_min = y_positions[-1] - box_height
    y_max = y_positions[0] + box_height
    if outcome_split:
        y_min = min(y_min, split_y - split_box_height / 2 - 0.9)
    ax.set_xlim(min_x - x_padding, max_x + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

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
