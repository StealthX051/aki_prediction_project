"""Shared publication appearance; the adjacent JSON is the single editable profile.

The same two files are copied into a portable workspace's 05_CODE directory.
Colors encode a named variable: input set, fold, or signed local contribution.
A figure's layout and scientific values remain the responsibility of its renderer.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

PROFILE_PATH = Path(__file__).with_suffix('.json')
STYLE = json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
STYLE_SHA256 = hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest()

def color(name: str) -> str:
    return STYLE['colors'][name]

def _line(value):
    return (value[0], tuple(value[1])) if isinstance(value, list) else value

INPUT_ORDER = tuple(STYLE['inputs'])
INPUT_COLORS = {key: color(value['color']) for key, value in STYLE['inputs'].items()}
INPUT_STYLES = {key: _line(value['linestyle']) for key, value in STYLE['inputs'].items()}
INPUT_MARKERS = {key: value['marker'] for key, value in STYLE['inputs'].items()}
INPUT_LABELS = {key: value['label'] for key, value in STYLE['inputs'].items()}
FAMILY_COLORS = {key: color(value) for key, value in STYLE['family_colors'].items()}
FEATURE_SET_COLORS = {key: color(value) for key, value in STYLE['feature_set_colors'].items()}
FEATURE_SET_COLORS.update(INPUT_COLORS)
FALLBACK_COLORS = tuple(color(value) for value in STYLE['fallback_colors'])
BRANCHES = STYLE['branches']
FOLD_COLORS = tuple(color(value['color']) for value in STYLE['folds'])
FOLD_STYLES = tuple(_line(value['linestyle']) for value in STYLE['folds'])
FOLD_MARKERS = tuple(value['marker'] for value in STYLE['folds'])
FONT_FAMILY = STYLE['figure']['font_family']
WIDTH_MM = STYLE['figure']['width_mm']
MIN_FONT_PT = STYLE['figure']['minimum_font_pt']
DEFAULT_FIGURE_FORMATS = tuple(STYLE['export']['default_report_formats'])
PNG_DPI = STYLE['export']['png_dpi']
TIFF_DPI = STYLE['export']['tiff_dpi']
JPG_DPI = STYLE['export']['jpg_dpi']
JPEG_QUALITY = STYLE['export']['jpg_quality']
TIFF_COMPRESSION = STYLE['export']['tiff_compression']
WORD_FONT = STYLE['word']['font_family']
WORD_BODY_PT = STYLE['word']['body_pt']
WORD_TABLE_PT = STYLE['word']['table_pt']
CAPTION_PT = STYLE['word']['caption_pt']
REFERENCE_STYLE = _line(STYLE['figure']['reference_linestyle'])
CURVE_WIDTH = STYLE['figure']['curve_width_pt']
GRID_WIDTH = STYLE['figure']['grid_width_pt']
REFERENCE_WIDTH = STYLE['figure']['reference_width_pt']


def mpl_style(profile: str = 'native', **overrides):
    """Default typography and neutral marks; layout-specific choices stay local."""
    from cycler import cycler
    figure = STYLE['figure']
    sizes = dict(figure)
    if profile == 'report':
        sizes.update(STYLE['report'])
    elif profile != 'native':
        raise ValueError(f'Unknown figure profile: {profile}')
    result = {
        'font.family': FONT_FAMILY, 'font.size': sizes['body_pt'],
        'axes.labelsize': sizes['axis_label_pt'], 'axes.titlesize': sizes['title_pt'],
        'axes.titleweight': 'normal', 'axes.labelcolor': color('ink'),
        'xtick.labelsize': sizes['tick_pt'], 'ytick.labelsize': sizes['tick_pt'],
        'legend.fontsize': sizes['legend_pt'], 'legend.frameon': False,
        'axes.prop_cycle': cycler(color=FALLBACK_COLORS),
        'text.color': color('ink'), 'axes.edgecolor': color('ink'),
        'xtick.color': color('ink'), 'ytick.color': color('ink'),
        'axes.linewidth': figure['spine_width_pt'], 'lines.linewidth': figure['curve_width_pt'],
        'xtick.major.width': figure['spine_width_pt'], 'ytick.major.width': figure['spine_width_pt'],
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'grid.color': color('grid'), 'grid.linewidth': figure['grid_width_pt'],
        'grid.alpha': 1, 'grid.linestyle': '-',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    }
    result.update(overrides)
    return result


@contextmanager
def style_context(profile: str = 'native', **overrides):
    import matplotlib as mpl
    with mpl.rc_context(mpl_style(profile, **overrides)):
        yield


def apply_mpl_style(profile: str = 'native', **overrides):
    import matplotlib as mpl
    mpl.rcParams.update(mpl_style(profile, **overrides))


def apply_document_style(document, *, body_size=None):
    """Resolve Word's themed headings explicitly without changing bold data runs."""
    from docx.shared import Pt, RGBColor
    sizes = STYLE['word']
    for name, key in [('Normal', 'body_pt'), ('Title', 'title_pt'),
                      ('Heading 1', 'heading_1_pt'), ('Heading 2', 'heading_2_pt'),
                      ('Heading 3', 'heading_3_pt'), ('Caption', 'caption_pt')]:
        style = document.styles[name]
        style.font.name = WORD_FONT
        style.font.size = Pt(body_size if name == 'Normal' and body_size is not None else sizes[key])
        style.font.color.rgb = RGBColor.from_string(color('ink').lstrip('#'))
        # Word theme attributes otherwise override explicit Arial in some templates.
        from docx.oxml.ns import qn
        fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
        for attribute in ('asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'cstheme', 'csTheme'):
            fonts.attrib.pop(qn('w:' + attribute), None)
        tint = style.element.rPr.find(qn('w:color'))
        if tint is not None:
            for attribute in ('themeColor', 'themeTint', 'themeShade'):
                tint.attrib.pop(qn('w:' + attribute), None)
    return document
