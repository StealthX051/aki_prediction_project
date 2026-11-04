"""Shared style edits propagate to plots and editable Word without changing values."""
from pathlib import Path
import importlib.util
import json

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import matplotlib as mpl
import pandas as pd

from reporting import publication_style as style
from reporting.figure_rendering import feature_set_color, report_figure_style_context
from reporting.manuscript_assets import write_docx_sections


def test_profile_edit_controls_colors_fonts_and_exports_in_a_fresh_process_module(tmp_path):
    source = Path(style.__file__)
    profile = json.loads(style.PROFILE_PATH.read_text())
    profile['colors']['blue'] = '#123456'
    profile['colors']['ink'] = '#343434'
    profile['figure']['font_family'] = 'DejaVu Serif'
    profile['export']['jpg_dpi'] = 240
    module_path = tmp_path / 'publication_style.py'
    module_path.write_text(source.read_text())
    module_path.with_suffix('.json').write_text(json.dumps(profile))
    spec = importlib.util.spec_from_file_location('changed_publication_style', module_path)
    changed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(changed)
    assert changed.INPUT_COLORS['preop_only'] == '#123456'
    assert changed.FEATURE_SET_COLORS['preop_only'] == '#123456'
    assert changed.FOLD_COLORS[0] == '#123456'
    assert changed.mpl_style()['font.family'] == 'DejaVu Serif'
    assert changed.mpl_style()['text.color'] == '#343434'
    assert changed.JPG_DPI == 240


def test_report_context_uses_shared_encodings_and_restores_global_settings():
    before = mpl.rcParams.copy()
    with report_figure_style_context():
        assert mpl.rcParams['axes.labelsize'] == style.STYLE['report']['axis_label_pt']
        assert mpl.rcParams['text.color'] == style.color('ink')
        for key, color in style.INPUT_COLORS.items():
            assert feature_set_color(key) == color
    assert mpl.rcParams['axes.labelsize'] == before['axes.labelsize']
    assert mpl.rcParams['text.color'] == before['text.color']
    import matplotlib.pyplot as plt
    with style.style_context():
        figure, axes = plt.subplots()
        try:
            lines = [axes.plot([0, 1], [i, i + 1])[0] for i in range(3)]
            assert [line.get_color() for line in lines] == [style.INPUT_COLORS[key] for key in style.INPUT_ORDER]
        finally:
            plt.close(figure)


def test_word_style_resolves_theme_fonts_and_preserves_bold_metric_values(tmp_path):
    frame = pd.DataFrame({'Model':['EBM', 'LR'], 'AP':['0.350 (0.285–0.427)', '0.400 (0.300–0.500)']})
    path = write_docx_sections(tmp_path/'table.docx', title='Core performance',
        sections=[{'heading':'Core performance', 'dataframe':frame,
                   'styles':{0:{'AP':{'bold':True}}}}])
    document = Document(path)
    for name in ['Normal', 'Title', 'Heading 1', 'Heading 2']:
        current = document.styles[name]
        assert current.font.name == style.WORD_FONT
        assert str(current.font.color.rgb) == style.color('ink').lstrip('#').upper()
        rpr = current.element.rPr
        fonts = rpr.find(qn('w:rFonts'))
        assert fonts is not None
        assert fonts.get(qn('w:asciiTheme')) is None
        assert fonts.get(qn('w:hAnsiTheme')) is None
    assert document.styles['Normal'].font.size.pt == style.WORD_TABLE_PT
    table = document.tables[0]
    assert 'w:tblHeader' in table.rows[0]._tr.xml
    assert all('w:cantSplit' in row._tr.xml for row in table.rows)
    assert len(table.rows) == len(frame) + 1
    assert table.cell(1, 0).paragraphs[0].alignment == WD_ALIGN_PARAGRAPH.LEFT
    assert table.cell(1, 1).paragraphs[0].alignment == WD_ALIGN_PARAGRAPH.RIGHT
    assert table.cell(1, 1).paragraphs[0].paragraph_format.space_after.pt == 0
    cell = table.cell(1,1)
    assert cell.text == frame.iloc[0,1]
    assert cell.paragraphs[0].runs[0].bold is True
    assert cell.paragraphs[0].runs[0].font.name == style.WORD_FONT
