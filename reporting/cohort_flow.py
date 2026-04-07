"""Create a cohort construction flow diagram from saved counts.

The flow diagram visualizes how many cases remain after each filtering step in
``data_preparation.step_01_cohort_construction``. The module accepts structured
counts saved to JSON (for example by persisting print statements from
``step_01``) and renders a Graphviz-first CONSORT-style flow chart to
``results/figures``. When Graphviz is unavailable, the module falls back to a
Matplotlib renderer so `.svg`/`.png` outputs still succeed.

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
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Any, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from artifact_paths import enforce_storage_policy, get_paper_dir, get_results_dir
from reporting.display_dictionary import DisplayDictionary, load_display_dictionary
from reporting.figure_rendering import mirror_figure_variants, report_figure_style_context, save_report_figure_bundle
from reporting.manuscript_assets import PNG_DPI, refresh_paper_bundle

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = get_results_dir(PROJECT_ROOT)
PAPER_DIR = get_paper_dir(PROJECT_ROOT)
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
        "filter_preop_cr": "Baseline Creatinine Eligibility",
        "ensure_sample_independence": "Subject Selection",
        "filter_postop_cr": "Outcome Data Availability",
        "add_aki_label": "AKI label derived",
    }

    removal_reason_map = {
        "mandatory_columns": "Missing mandatory data fields",
        "waveforms": "Incomplete waveform data",
        "Exclude ASA V/VI": "ASA class V/VI excluded",
        "filter_preop_cr": "Baseline creatinine > 4.0 mg/dL",
        "ensure_sample_independence": "Exclusion of repeat encounters",
        "filter_postop_cr": "Missing postoperative creatinine",
    }

    title_overrides = {
        "mandatory_columns": "EHR Record Screening",
        "waveforms": "High-fidelity Waveform Availability",
        "Exclude ASA V/VI": "ASA Class Exclusion",
        "filter_preop_cr": "Baseline Creatinine Eligibility",
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

    def _normalize_custom_filter_entries(entries: Sequence[Any]) -> List[Dict[str, Any]]:
        normalized_entries: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                logger.warning("Skipping unrecognized custom filter entry: %s", entry)
                continue

            raw_name = str(entry.get("name") or "Custom filter")
            count = _extract_count(entry)
            if count is None:
                continue

            count_before = entry.get("count_before")
            try:
                count_before_value = int(count_before) if count_before is not None else None
            except (TypeError, ValueError):
                count_before_value = None

            if raw_name in title_overrides:
                title = title_overrides[raw_name]
            else:
                title = str(
                    entry.get("label")
                    or filter_label_map.get(raw_name)
                    or raw_name.replace("_", " ").title()
                )

            normalized_entries.append(
                {
                    "raw_name": raw_name,
                    "title": title,
                    "count": count,
                    "count_before": count_before_value,
                    "removal_reason": removal_reason_map.get(raw_name),
                }
            )

        return normalized_entries

    def _order_custom_filter_entries(
        entries: Sequence[Dict[str, Any]],
        *,
        start_count: Optional[int],
    ) -> List[Dict[str, Any]]:
        remaining = list(entries)
        ordered: List[Dict[str, Any]] = []
        current_count = start_count

        while remaining:
            next_idx = None
            if current_count is not None:
                for idx, entry in enumerate(remaining):
                    if entry.get("count_before") == current_count:
                        next_idx = idx
                        break

            if next_idx is None:
                next_idx = 0
                entry = remaining[next_idx]
                if current_count is not None and entry.get("count_before") is not None:
                    logger.warning(
                        "Could not align custom filter %s using count_before=%s after count=%s; "
                        "keeping input order.",
                        entry["raw_name"],
                        entry.get("count_before"),
                        current_count,
                    )

            entry = remaining.pop(next_idx)
            ordered.append(entry)
            current_count = entry["count"]

        return ordered

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
    else:
        final_waveform_count = None

    custom_filters = raw_counts.get("custom_filters") or []
    if isinstance(custom_filters, Mapping):
        custom_filters = [
            {"name": key, "count": value} for key, value in custom_filters.items()
        ]

    custom_filter_entries = _normalize_custom_filter_entries(custom_filters)
    pre_waveform_filters: List[Dict[str, Any]] = []
    post_waveform_filters: List[Dict[str, Any]] = []
    for entry in custom_filter_entries:
        if (
            final_waveform_count is not None
            and entry.get("count_before") is not None
            and entry["count_before"] > final_waveform_count
        ):
            pre_waveform_filters.append(entry)
        else:
            post_waveform_filters.append(entry)

    previous_stage_count = stages[-1].count if stages else None
    for entry in _order_custom_filter_entries(pre_waveform_filters, start_count=previous_stage_count):
        stages.append(
            CohortStage(
                title=entry["title"],
                count=entry["count"],
                removal_reason=entry["removal_reason"],
            )
        )

    if final_waveform_count is not None:
        stages.append(
            CohortStage(
                title=title_overrides.get("waveforms", "Waveform availability"),
                count=final_waveform_count,
                detail=detail,
                removal_reason=removal_reason_map.get("waveforms"),
            )
        )

    previous_stage_count = stages[-1].count if stages else None
    for entry in _order_custom_filter_entries(post_waveform_filters, start_count=previous_stage_count):
        stages.append(
            CohortStage(
                title=entry["title"],
                count=entry["count"],
                removal_reason=entry["removal_reason"],
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


def _format_n(count: int) -> str:
    return f"{count:,}"


def _wrap_lines(lines: Sequence[str], *, width: int) -> List[str]:
    wrapped: List[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        if line.startswith("- "):
            fragments = textwrap.wrap(line[2:], width=width, subsequent_indent="  ")
            if not fragments:
                wrapped.append(line)
                continue
            wrapped.append(f"- {fragments[0]}")
            wrapped.extend(fragments[1:])
            continue
        wrapped.extend(textwrap.wrap(line, width=width) or [line])
    return wrapped


def _escape_dot(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _dot_label(lines: Sequence[str], *, left_aligned: bool) -> str:
    escaped = [_escape_dot(line) for line in lines]
    if left_aligned:
        return "\\l".join(escaped) + "\\l"
    return "\\n".join(escaped)


def _stage_node_lines(stage: CohortStage) -> List[str]:
    lines = _wrap_lines([stage.title], width=28)
    if stage.detail:
        lines.extend(_wrap_lines([stage.detail], width=34))
    lines.append(f"n = {_format_n(stage.count)}")
    return lines


def _exclusion_node_lines(stage: CohortStage) -> List[str]:
    if not stage.removed:
        return []

    heading = stage.removal_reason or stage.title
    lines = [f"{heading} (n = {_format_n(stage.removed)})"]
    if "waveform" in stage.title.lower() and stage.detail:
        lines.append(f"- {stage.detail}")
    return _wrap_lines(lines, width=34)


def _split_node_lines(label: str, count: int) -> List[str]:
    return _wrap_lines([label], width=24) + [f"n = {_format_n(count)}"]


def _split_labels(outcome_split: OutcomeSplit) -> Tuple[str, str]:
    label_lower = outcome_split.label.lower()
    if "aki" in label_lower:
        return "No AKI", "Acute Kidney Injury (AKI)"
    return f"{outcome_split.label} - False", f"{outcome_split.label} - True"


def _graph_attrs(title: Optional[str]) -> str:
    attrs = [
        'rankdir="TB"',
        'splines=polyline',
        'newrank=true',
        'ordering="out"',
        'pad="0.18"',
        'nodesep="0.55"',
        'ranksep="0.9"',
        'fontname="Times New Roman"',
    ]
    if title:
        attrs.extend(
            [
                'labelloc="t"',
                'labeljust="c"',
                'fontsize=20',
                f'label="{_escape_dot(title)}"',
            ]
        )
    return ", ".join(attrs)


def _cohort_flow_dot(stages: Sequence[CohortStage], outcome_split: Optional[OutcomeSplit], title: Optional[str]) -> str:
    stage_ids = [f"stage_{idx}" for idx in range(len(stages))]
    exclusion_ids: List[Optional[str]] = []
    lines = [
        "digraph cohort_flow {",
        f"  graph [{_graph_attrs(title)}];",
        '  node [shape=box, style="rounded,filled", fontname="Times New Roman", fontsize=12, penwidth=1.5, color="#667d93", fillcolor="#f8fafc", margin="0.20,0.14"];',
        '  edge [color="#667d93", penwidth=1.5, arrowsize=0.75];',
    ]

    for idx, stage in enumerate(stages):
        stage_id = stage_ids[idx]
        fillcolor = "#f2f7fb" if idx == len(stages) - 1 else "#f8fafc"
        lines.append(
            f'  {stage_id} [label="{_dot_label(_stage_node_lines(stage), left_aligned=False)}", '
            f'width=4.0, height=1.0, fillcolor="{fillcolor}"];'
        )

        exclusion_id = None
        if stage.removed:
            exclusion_id = f"excluded_{idx}"
            lines.append(
                f'  {exclusion_id} [label="{_dot_label(_exclusion_node_lines(stage), left_aligned=True)}", '
                'width=3.55, fontsize=10.5, style="rounded,filled,dashed", '
                'color="#7b8b99", fillcolor="#fbfcfd", margin="0.16,0.12"];'
            )
        exclusion_ids.append(exclusion_id)

    for idx in range(len(stage_ids) - 1):
        lines.append(f"  {stage_ids[idx]}:s -> {stage_ids[idx + 1]}:n;")

    previous_exclusion_id: Optional[str] = None
    for idx, exclusion_id in enumerate(exclusion_ids):
        if not exclusion_id:
            continue
        lines.append(
            f'  {stage_ids[idx]}:e -> {exclusion_id}:w [style=dashed, color="#7b8b99", '
            'constraint=false, minlen=2];'
        )
        lines.append(f"  {{ rank=same; {stage_ids[idx]}; {exclusion_id}; }}")
        if previous_exclusion_id:
            lines.append(f"  {previous_exclusion_id} -> {exclusion_id} [style=invis, weight=20];")
        previous_exclusion_id = exclusion_id

    if outcome_split:
        negative_label, positive_label = _split_labels(outcome_split)
        lines.extend(
            [
                f'  final_negative [label="{_dot_label(_split_node_lines(negative_label, outcome_split.false_count), left_aligned=False)}", width=3.0, height=0.95, fillcolor="#f8fafc"];',
                f'  final_positive [label="{_dot_label(_split_node_lines(positive_label, outcome_split.true_count), left_aligned=False)}", width=3.0, height=0.95, fillcolor="#edf5fb"];',
                f'  {stage_ids[-1]}:s -> final_negative:n [minlen=1];',
                f'  {stage_ids[-1]}:s -> final_positive:n [minlen=1];',
                "  { rank=same; final_negative; final_positive; }",
                "  final_negative -> final_positive [style=invis, weight=20];",
            ]
        )

    lines.append("}")
    return "\n".join(lines) + "\n"


def _box_height(lines: Sequence[str], *, line_height: float, minimum: float) -> float:
    return max(minimum, 0.045 + len(lines) * line_height)


def _draw_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: Sequence[str],
    facecolor: str,
    edgecolor: str,
    fontsize: float,
    dashed: bool = False,
    centered: bool = True,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.35,
        linestyle=(0, (4, 2)) if dashed else "solid",
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2 if centered else x + 0.025,
        y + height / 2,
        "\n".join(lines),
        ha="center" if centered else "left",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if centered else "normal",
        linespacing=1.35,
        color="#1f2933",
    )


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#607286", "shrinkA": 2, "shrinkB": 2},
    )


def _draw_cohort_flow_fallback(
    stages: Sequence[CohortStage],
    outcome_split: Optional[OutcomeSplit],
    title: Optional[str],
) -> plt.Figure:
    stage_lines = [_stage_node_lines(stage) for stage in stages]
    exclusion_lines = [_exclusion_node_lines(stage) for stage in stages]
    split_lines = None
    if outcome_split:
        negative_label, positive_label = _split_labels(outcome_split)
        split_lines = (
            _split_node_lines(negative_label, outcome_split.false_count),
            _split_node_lines(positive_label, outcome_split.true_count),
        )

    fig, ax = plt.subplots(figsize=(11.0, max(8.2, 2.2 + len(stages) * 1.55)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    main_x = 0.08
    main_w = 0.42
    side_x = 0.61
    side_w = 0.29
    top_y = 0.86
    bottom_y = 0.22 if outcome_split else 0.08
    stage_centers = (
        [0.50]
        if len(stages) == 1
        else list(reversed([bottom_y + (top_y - bottom_y) * idx / (len(stages) - 1) for idx in range(len(stages))]))
    )

    stage_boxes: List[tuple[float, float, float, float]] = []
    exclusion_boxes: List[Optional[tuple[float, float, float, float]]] = []
    for idx, lines in enumerate(stage_lines):
        height = _box_height(lines, line_height=0.035, minimum=0.12)
        center_y = stage_centers[idx]
        y = max(0.05, center_y - height / 2)
        stage_boxes.append((main_x, y, main_w, height))

        if exclusion_lines[idx]:
            exclusion_height = _box_height(exclusion_lines[idx], line_height=0.028, minimum=0.14)
            exclusion_y = max(0.05, center_y - exclusion_height / 2)
            exclusion_boxes.append((side_x, exclusion_y, side_w, exclusion_height))
        else:
            exclusion_boxes.append(None)

    with report_figure_style_context():
        for idx, stage in enumerate(stages):
            x, y, width, height = stage_boxes[idx]
            fillcolor = "#f2f7fb" if idx == len(stages) - 1 else "#f8fafc"
            _draw_box(
                ax,
                x=x,
                y=y,
                width=width,
                height=height,
                lines=stage_lines[idx],
                facecolor=fillcolor,
                edgecolor="#607286",
                fontsize=10.2,
            )

            exclusion_box = exclusion_boxes[idx]
            if exclusion_box is not None:
                ex_x, ex_y, ex_width, ex_height = exclusion_box
                _draw_box(
                    ax,
                    x=ex_x,
                    y=ex_y,
                    width=ex_width,
                    height=ex_height,
                    lines=exclusion_lines[idx],
                    facecolor="#fbfcfd",
                    edgecolor="#7b8b99",
                    fontsize=8.7,
                    dashed=True,
                    centered=False,
                )
                stage_mid_y = y + height / 2
                ax.plot(
                    [x + width, ex_x],
                    [stage_mid_y, stage_mid_y],
                    color="#7b8b99",
                    lw=1.2,
                    linestyle=(0, (3, 2)),
                )

        for idx in range(len(stage_boxes) - 1):
            x, y, width, _height = stage_boxes[idx]
            next_x, next_y, next_width, next_height = stage_boxes[idx + 1]
            _draw_arrow(
                ax,
                (x + width / 2, y),
                (next_x + next_width / 2, next_y + next_height),
            )

        if outcome_split and split_lines is not None:
            branch_height = max(
                _box_height(split_lines[0], line_height=0.035, minimum=0.11),
                _box_height(split_lines[1], line_height=0.035, minimum=0.11),
            )
            left_box = (0.07, 0.04, 0.34, branch_height)
            right_box = (0.45, 0.04, 0.34, branch_height)
            _draw_box(
                ax,
                x=left_box[0],
                y=left_box[1],
                width=left_box[2],
                height=left_box[3],
                lines=split_lines[0],
                facecolor="#f8fafc",
                edgecolor="#607286",
                fontsize=9.8,
            )
            _draw_box(
                ax,
                x=right_box[0],
                y=right_box[1],
                width=right_box[2],
                height=right_box[3],
                lines=split_lines[1],
                facecolor="#eef5fb",
                edgecolor="#607286",
                fontsize=9.8,
            )

            parent_x, parent_y, parent_w, _parent_h = stage_boxes[-1]
            split_y = left_box[1] + left_box[3] + 0.05
            parent_center = parent_x + parent_w / 2
            left_center = left_box[0] + left_box[2] / 2
            right_center = right_box[0] + right_box[2] / 2
            ax.plot([parent_center, parent_center], [parent_y, split_y], color="#607286", lw=1.5)
            ax.plot([left_center, right_center], [split_y, split_y], color="#607286", lw=1.5)
            _draw_arrow(ax, (left_center, split_y), (left_center, left_box[1] + left_box[3]))
            _draw_arrow(ax, (right_center, split_y), (right_center, right_box[1] + right_box[3]))

        if title:
            ax.text(
                0.5,
                0.985,
                title,
                ha="center",
                va="top",
                fontsize=14,
                fontweight="bold",
                color="#1f2933",
            )
    return fig


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

    enforce_storage_policy({"paper_figures_dir": output_dir})
    output_dir.mkdir(parents=True, exist_ok=True)

    base_path = (output_dir / output_name).with_suffix("")
    dot_path = base_path.with_suffix(".dot")
    dot_path.write_text(_cohort_flow_dot(stages, outcome_split, title), encoding="utf-8")

    saved_paths = [dot_path]
    requested_formats: List[str] = []
    seen_formats = set()
    for fmt in formats:
        normalized = str(fmt).lower().lstrip(".")
        if normalized == "dot" or normalized in seen_formats:
            continue
        requested_formats.append(normalized)
        seen_formats.add(normalized)

    dot_binary = shutil.which("dot")
    if dot_binary is not None:
        for fmt in requested_formats:
            output_path = base_path.with_suffix(f".{fmt}")
            command = [dot_binary, f"-T{fmt}", str(dot_path), "-o", str(output_path)]
            if fmt == "png":
                command.insert(1, f"-Gdpi={PNG_DPI}")
            subprocess.run(command, check=True)
            saved_paths.append(output_path)
            logger.info("Saved cohort flow diagram to %s", output_path)
        mirror_figure_variants(base_path, formats=requested_formats)
    elif requested_formats:
        fig = _draw_cohort_flow_fallback(stages, outcome_split, title)
        try:
            saved_paths.extend(
                save_report_figure_bundle(
                    fig,
                    base_path,
                    formats=requested_formats,
                    close=True,
                    mirror_to_primary=True,
                )
            )
        finally:
            plt.close(fig)
        logger.warning("Graphviz 'dot' binary not found; used Matplotlib fallback for %s.", base_path)

    refresh_paper_bundle(PAPER_DIR)
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
