from __future__ import annotations

from contextlib import contextmanager
import hashlib
import shutil
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt

from reporting.manuscript_assets import PNG_DPI, save_figure_bundle


PRIMARY_FIGURE_SUBDIR = "primary_figures"
DEFAULT_FIGURE_FORMATS: tuple[str, ...] = ("svg", "png")

_FEATURE_SET_COLORS = {
    "preop_only": "#2d6ba3",
    "all_waveforms": "#bf3b3b",
    "preop_and_all_waveforms": "#2f855a",
    "awp_only": "#a06b38",
    "co2_only": "#bc5090",
    "ecg_only": "#4c956c",
    "pleth_only": "#6a4c93",
    "monitors_only": "#52606d",
    "ventilator_only": "#d08c24",
    "preop_and_awp": "#8f5d2f",
    "preop_and_co2": "#a23b72",
    "preop_and_ecg": "#127475",
    "preop_and_pleth": "#4f5d95",
    "preop_and_all_minus_awp": "#7b8c38",
    "preop_and_all_minus_co2": "#a64b00",
    "preop_and_all_minus_ecg": "#8c564b",
    "preop_and_all_minus_pleth": "#5f0f40",
}
_FALLBACK_COLORS = (
    "#2d6ba3",
    "#bf3b3b",
    "#2f855a",
    "#6a4c93",
    "#a06b38",
    "#127475",
    "#bc5090",
    "#52606d",
    "#d08c24",
    "#8c564b",
)


def feature_set_color(feature_set: str) -> str:
    color = _FEATURE_SET_COLORS.get(str(feature_set))
    if color:
        return color
    token = hashlib.md5(str(feature_set).encode("utf-8")).hexdigest()
    return _FALLBACK_COLORS[int(token[:8], 16) % len(_FALLBACK_COLORS)]


@contextmanager
def report_figure_style_context() -> Iterator[None]:
    import matplotlib as mpl

    rc_params = {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": "#1f2933",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": "#1f2933",
        "axes.edgecolor": "#52606d",
        "axes.linewidth": 1.05,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.color": "#cbd2d9",
        "grid.alpha": 0.22,
        "grid.linewidth": 0.8,
        "grid.linestyle": "--",
    }
    with mpl.rc_context(rc=rc_params):
        yield


def primary_figure_dir(figures_dir: Path) -> Path:
    return figures_dir / PRIMARY_FIGURE_SUBDIR


def mirror_figure_variants(
    destination: Path,
    *,
    formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
    primary_subdir: str = PRIMARY_FIGURE_SUBDIR,
) -> list[Path]:
    base = destination.with_suffix("") if destination.suffix else destination
    if base.parent.name == primary_subdir:
        return []

    mirror_dir = base.parent / primary_subdir
    mirror_dir.mkdir(parents=True, exist_ok=True)
    mirrored: list[Path] = []
    for fmt in formats:
        normalized = str(fmt).lower().lstrip(".")
        source = base.with_suffix(f".{normalized}")
        if not source.exists():
            continue
        target = mirror_dir / source.name
        shutil.copy2(source, target)
        mirrored.append(target)
    return mirrored


def save_report_figure_bundle(
    fig: plt.Figure,
    destination: Path,
    *,
    formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
    png_dpi: int = PNG_DPI,
    close: bool = True,
    mirror_to_primary: bool = False,
) -> list[Path]:
    saved = save_figure_bundle(
        fig,
        destination,
        formats=formats,
        png_dpi=png_dpi,
        close=False,
    )
    if mirror_to_primary:
        mirror_figure_variants(destination, formats=formats)
    if close:
        plt.close(fig)
    return saved

