from __future__ import annotations

from contextlib import contextmanager
import hashlib
import shutil
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt

from reporting.manuscript_assets import DEFAULT_FIGURE_FORMATS, PNG_DPI, save_figure_bundle
from reporting.publication_style import FEATURE_SET_COLORS, FALLBACK_COLORS, mpl_style


PRIMARY_FIGURE_SUBDIR = "primary_figures"

_FEATURE_SET_COLORS = FEATURE_SET_COLORS
_FALLBACK_COLORS = FALLBACK_COLORS


def feature_set_color(feature_set: str) -> str:
    color = _FEATURE_SET_COLORS.get(str(feature_set))
    if color:
        return color
    token = hashlib.md5(str(feature_set).encode("utf-8")).hexdigest()
    return _FALLBACK_COLORS[int(token[:8], 16) % len(_FALLBACK_COLORS)]


@contextmanager
def report_figure_style_context() -> Iterator[None]:
    import matplotlib as mpl

    rc_params = mpl_style('report', **{'axes.grid': True})
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

