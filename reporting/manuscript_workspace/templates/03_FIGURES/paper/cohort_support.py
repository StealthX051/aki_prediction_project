from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '05_CODE'))
import publication_style as ps
from dataclasses import dataclass
from typing import Optional,Sequence,List,Tuple
from contextlib import contextmanager
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
@contextmanager
def report_figure_style_context():
    with ps.style_context():
        yield
@dataclass(frozen=True)
class CohortStage:
    """Normalized representation of a cohort filtering stage."""
    title: str
    count: int
    detail: Optional[str] = None
    removed: Optional[int] = None
    removal_reason: Optional[str] = None
    n_patients: Optional[int] = None

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

def _format_n(count: int) -> str:
    return f'{count:,}'

def _wrap_lines(lines: Sequence[str], *, width: int) -> List[str]:
    wrapped: List[str] = []
    for line in lines:
        if not line:
            wrapped.append('')
            continue
        if line.startswith('- '):
            fragments = textwrap.wrap(line[2:], width=width, subsequent_indent='  ')
            if not fragments:
                wrapped.append(line)
                continue
            wrapped.append(f'- {fragments[0]}')
            wrapped.extend(fragments[1:])
            continue
        wrapped.extend(textwrap.wrap(line, width=width) or [line])
    return wrapped

def _stage_node_lines(stage: CohortStage) -> List[str]:
    lines = _wrap_lines([stage.title], width=28)
    if stage.detail:
        lines.extend(_wrap_lines([stage.detail], width=34))
    lines.append(f'n = {_format_n(stage.count)} operations')
    if stage.n_patients is not None:
        lines.append(f'{_format_n(stage.n_patients)} unique patients')
    return lines

def _exclusion_node_lines(stage: CohortStage) -> List[str]:
    if not stage.removed:
        return []
    heading = stage.removal_reason or stage.title
    lines = [f'{heading} (n = {_format_n(stage.removed)})']
    if 'waveform' in stage.title.lower() and stage.detail:
        lines.append(f'- {stage.detail}')
    return _wrap_lines(lines, width=34)

def _split_node_lines(label: str, count: int) -> List[str]:
    return _wrap_lines([label], width=24) + [f'n = {_format_n(count)}']

def _split_labels(outcome_split: OutcomeSplit) -> Tuple[str, str]:
    label_lower = outcome_split.label.lower()
    if 'aki' in label_lower:
        return ('No AKI', 'Acute Kidney Injury (AKI)')
    return (f'{outcome_split.label} - False', f'{outcome_split.label} - True')

def _draw_box(ax: plt.Axes, *, x: float, y: float, width: float, height: float, lines: Sequence[str], facecolor: str, edgecolor: str, fontsize: float, dashed: bool=False, centered: bool=True) -> None:
    patch = FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.04,rounding_size=0.08', linewidth=1.35, linestyle=(0, (4, 2)) if dashed else 'solid', edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(patch)
    ax.text(x + width / 2 if centered else x + 0.15, y + height / 2, '\n'.join(lines), ha='center' if centered else 'left', va='center', fontsize=fontsize, fontweight='bold' if centered else 'normal', linespacing=1.35, color='#1f2933')

def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '-|>', 'lw': 1.5, 'color': '#607286', 'shrinkA': 2, 'shrinkB': 2})

def _draw_cohort_flow_fallback(stages: Sequence[CohortStage], outcome_split: Optional[OutcomeSplit], title: Optional[str], *, publication: bool=False) -> plt.Figure:
    """Render the retained spine and exclusions; publication uses a native 174 mm canvas."""
    stage_lines = [_stage_node_lines(stage) for stage in stages]
    exclusion_lines = [_exclusion_node_lines(stage) for stage in stages]
    if publication:
        exclusion_lines = [_wrap_lines([stage.removal_reason or stage.title], width=30) + [f'n = {_format_n(stage.removed)}'] if stage.removed else [] for stage in stages]
    split_lines = None
    if outcome_split:
        negative_label, positive_label = _split_labels(outcome_split)
        split_lines = (_split_node_lines(negative_label, outcome_split.false_count), _split_node_lines(positive_label, outcome_split.true_count))

    def text_height(lines, fontsize):
        return max(0.65, len(lines) * fontsize * 1.35 / 72 + 0.36)
    main_heights = [text_height(lines, 10.2) for lines in stage_lines]
    side_heights = [text_height(lines, 8.7) if lines else 0 for lines in exclusion_lines]
    row_heights = [max(main, side) for main, side in zip(main_heights, side_heights)]
    branch_height = max((text_height(lines, 9.8) for lines in split_lines)) if split_lines else 0
    row_gap = 0.24 if publication else 0.38
    bottom_margin = 0.35
    split_space = branch_height + 0.65 if split_lines else 0
    title_space = 0.7 if title else 0.25
    figure_height = sum(row_heights) + max(0, len(stages) - 1) * row_gap + bottom_margin + split_space + title_space
    figure_width = ps.WIDTH_MM / 25.4 if publication else 10.6
    fig = plt.figure(figsize=(figure_width, figure_height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, figure_width)
    ax.set_ylim(0, figure_height)
    ax.axis('off')
    scale = figure_width / 10.6
    main_x, main_w, side_x, side_w = [value * scale for value in (0.4, 5.0, 6.1, 4.1)]
    stage_boxes: List[tuple[float, float, float, float]] = []
    exclusion_boxes: List[Optional[tuple[float, float, float, float]]] = []
    row_top = figure_height - title_space
    for idx, height in enumerate(main_heights):
        center_y = row_top - row_heights[idx] / 2
        stage_boxes.append((main_x, center_y - height / 2, main_w, height))
        if exclusion_lines[idx]:
            exclusion_height = side_heights[idx]
            exclusion_boxes.append((side_x, center_y - exclusion_height / 2, side_w, exclusion_height))
        else:
            exclusion_boxes.append(None)
        row_top -= row_heights[idx] + row_gap
    exclusion_guides = []
    with report_figure_style_context():
        for idx, stage in enumerate(stages):
            x, y, width, height = stage_boxes[idx]
            fillcolor = '#f2f7fb' if idx == len(stages) - 1 else '#f8fafc'
            _draw_box(ax, x=x, y=y, width=width, height=height, lines=stage_lines[idx], facecolor=fillcolor, edgecolor='#607286', fontsize=10.2)
            exclusion_box = exclusion_boxes[idx]
            if exclusion_box is not None:
                ex_x, ex_y, ex_width, ex_height = exclusion_box
                _draw_box(ax, x=ex_x, y=ex_y, width=ex_width, height=ex_height, lines=exclusion_lines[idx], facecolor='#fbfcfd', edgecolor='#7b8b99', fontsize=8.7, dashed=True, centered=False)
                stage_mid_y = y + height / 2
                exclusion_guides.extend(ax.plot([x + width, ex_x], [stage_mid_y, stage_mid_y], color='#7b8b99', lw=ps.STYLE['figure']['curve_width_pt'], linestyle=(0, (3, 2))))
        for idx in range(len(stage_boxes) - 1):
            x, y, width, _height = stage_boxes[idx]
            next_x, next_y, next_width, next_height = stage_boxes[idx + 1]
            _draw_arrow(ax, (x + width / 2, y), (next_x + next_width / 2, next_y + next_height))
        if outcome_split and split_lines is not None:
            left_box = (0.4 * scale, bottom_margin, 4.3 * scale, branch_height)
            right_box = (5.4 * scale, bottom_margin, 4.3 * scale, branch_height)
            _draw_box(ax, x=left_box[0], y=left_box[1], width=left_box[2], height=left_box[3], lines=split_lines[0], facecolor='#f8fafc', edgecolor='#607286', fontsize=9.8)
            _draw_box(ax, x=right_box[0], y=right_box[1], width=right_box[2], height=right_box[3], lines=split_lines[1], facecolor='#eef5fb', edgecolor='#607286', fontsize=9.8)
            parent_x, parent_y, parent_w, _parent_h = stage_boxes[-1]
            split_y = left_box[1] + left_box[3] + 0.3
            parent_center = parent_x + parent_w / 2
            left_center = left_box[0] + left_box[2] / 2
            right_center = right_box[0] + right_box[2] / 2
            ax.plot([parent_center, parent_center], [parent_y, split_y], color='#607286', lw=1.5)
            ax.plot([left_center, right_center], [split_y, split_y], color='#607286', lw=1.5)
            _draw_arrow(ax, (left_center, split_y), (left_center, left_box[1] + left_box[3]))
            _draw_arrow(ax, (right_center, split_y), (right_center, right_box[1] + right_box[3]))
        if title:
            ax.text(figure_width / 2, figure_height - 0.18, title, ha='center', va='top', fontsize=14, fontweight='bold', color='#1f2933')
    if publication:
        ax.text(side_x + side_w / 2, stage_boxes[0][1] + stage_boxes[0][3] / 2, 'Excluded operations', ha='center', va='center', fontsize=9.5, color=ps.color('ink'))
        for patch in ax.patches:
            patch.set_facecolor('white')
            patch.set_edgecolor(ps.color('ink'))
            patch.set_linewidth(0.9)
            patch.set_linestyle('solid')
        for text in ax.texts:
            text.set_color(ps.color('ink'))
            text.set_fontweight('normal')
            if getattr(text, 'arrow_patch', None) is not None:
                text.arrow_patch.set_color(ps.color('ink'))
                text.arrow_patch.set_linewidth(1)
        for line in exclusion_guides:
            line.remove()
        for main, excluded in zip(stage_boxes, exclusion_boxes):
            if excluded is None:
                continue
            x, y, width, height = main
            ex_x, ex_y, ex_width, ex_height = excluded
            ax.annotate('', xy=(ex_x, ex_y + ex_height / 2), xytext=(x + width, y + height / 2), arrowprops={'arrowstyle': '-|>', 'lw': 1, 'color': ps.color('ink'), 'shrinkA': 2, 'shrinkB': 2})
        for text in ax.texts:
            if text.get_text() == '\n'.join(stage_lines[-1]):
                text.set_fontweight('bold')
    return fig
