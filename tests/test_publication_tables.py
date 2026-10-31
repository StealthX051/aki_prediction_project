from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from docx import Document

from reporting import manuscript_assets as assets
from results_recreation.results_analysis import prepare_metrics_for_display


def _scores():
    rows = []
    for model in ('Logistic regression', 'XGBoost', 'EBM'):
        for index, features in enumerate(('preop_only', 'all_waveforms', 'preop_and_all_waveforms')):
            row = dict(outcome='any_aki', branch='non_windowed', model_name=model,
                       feature_set=features, n=40, n_patients=40, prevalence=.25,
                       analysis_status='diagnostic', auprc=.3 + index * .01,
                       auroc=.6 + index * .02, brier=.2 - index * .01,
                       sensitivity=.5, specificity=.7, precision=.4, accuracy=.6, f1=.4,
                       delta_auprc=-.03 + index * .01)
            for metric in ('auprc', 'auroc', 'brier', 'delta_auprc'):
                row[metric + '_lower'] = row[metric] - .04
                row[metric + '_upper'] = row[metric] + .04
            rows.append(row)
    return pd.DataFrame(rows)


def test_core_paper_view_keeps_full_data_and_consistent_descriptive_styling(tmp_path):
    assets.export_core_comparisons(prepare_metrics_for_display(_scores()), tmp_path)
    base = tmp_path / 'manuscript_any_aki_main'
    raw = pd.read_csv(base.with_suffix('.csv'))
    assert {'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'Operations', 'Patients'} <= set(raw)
    html = base.with_suffix('.html').read_text()
    assert 'Average Precision' not in html  # concise AP label, expanded in the caption
    assert '>AP<' in html and 'average precision' in html
    assert '40 operations from 40 patients' in html
    assert 'not evidence of statistical superiority' in html
    assert 'diagnostic' in html and 'not definitive manuscript evidence' in html
    assert 'background-color' in html and 'font-weight: bold' in html
    document = Document(base.with_suffix('.docx'))
    assert [cell.text for cell in document.tables[0].rows[0].cells] == ['Model', 'Feature set', 'AP', 'AUROC', 'Brier', 'Δ AP']
    for row in document.tables[0].rows[1:]:
        best = row.cells[1].text == 'Preoperative + all waveforms'
        for cell in row.cells[2:5]:
            assert any(run.bold for paragraph in cell.paragraphs for run in paragraph.runs) == best
    assert any('40 operations from 40 patients' in paragraph.text for paragraph in document.paragraphs)
    gain = pd.read_csv(tmp_path.parent / 'figures/incremental_ap_any_aki_main.csv')
    assert gain.Model.tolist() == ['Logistic regression', 'XGBoost', 'EBM']
    assert gain.analysis_status.eq('diagnostic').all()
    assert (tmp_path.parent / 'figures/incremental_ap_any_aki_main.svg').exists()
    assert any(cell._tc.xpath('.//w:shd') for row in document.tables[0].rows[1:] for cell in row.cells)


def test_paper_pdf_text_fits_inside_cells_with_long_labels_and_signed_intervals():
    frame = pd.DataFrame({'Model': ['Logistic regression'] * 3,
                          'Feature set': ['Preoperative only', 'All waveforms', 'Preoperative + all waveforms'],
                          'AP': ['0.300 (0.200 to 0.400)'] * 3,
                          'AUROC': ['0.600 (0.500 to 0.700)'] * 3,
                          'Brier': ['0.200 (0.100 to 0.300)'] * 3,
                          'Δ AP': ['−0.030 (−0.070 to 0.010)'] * 3})
    class InspectPdf:
        def savefig(self, fig, **kwargs):
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            table = fig.axes[0].tables[0]
            for cell in table.get_celld().values():
                text = cell.get_text().get_window_extent(renderer)
                bounds = cell.get_window_extent(renderer)
                assert text.x0 >= bounds.x0 - 1 and text.x1 <= bounds.x1 + 1
                assert text.y0 >= bounds.y0 - 1 and text.y1 <= bounds.y1 + 1
    assets._render_pdf_table_page(InspectPdf(), 'Diagnostic model performance', frame,
                                 notes='95% patient-cluster intervals. Descriptive ranks only.')


def test_supplement_preserves_branch_identity_and_different_populations(tmp_path):
    scores = _scores().iloc[:2].copy()
    scores['feature_set'] = 'preop_and_pleth'
    scores['branch'] = ['non_windowed', 'windowed']
    scores['n_patients'] = [39, 38]
    scores['prevalence'] = [.1, .3]
    assets.export_core_comparisons(prepare_metrics_for_display(scores), tmp_path)
    base = tmp_path / 'manuscript_any_aki_supplement'
    html = base.with_suffix('.html').read_text()
    assert 'Whole case' in html and 'Windowed' in html
    assert 'Populations differ across rows' in html
    assert 'event prevalence 10.0%' not in html
    document = Document(base.with_suffix('.docx'))
    assert 'Branch' in [c.text for c in document.tables[0].rows[0].cells]


def test_wide_pdf_panels_repeat_complete_model_input_population_identity():
    frame = pd.DataFrame({'Model': ['XGBoost'], 'Branch': ['Windowed'],
                          'Feature set': ['Clinical + waveforms'], 'Outcome': ['AKI'],
                          **{f'Metric {i}': [.123] for i in range(12)}})
    panels = []
    class InspectPdf:
        def savefig(self, fig, **kwargs):
            columns = [' '.join(cell.get_text().get_text().split())
                       for (row, _), cell in fig.axes[0].tables[0].get_celld().items() if row == 0]
            panels.append(columns)
    assets._render_pdf_table_page(InspectPdf(), 'Full metrics', frame)
    assert len(panels) == 4
    assert all({'Model', 'Branch', 'Feature set', 'Outcome'} <= set(columns) for columns in panels)
    assert {f'Metric {i}' for i in range(12)} <= set(sum(panels, []))


def test_publication_comparisons_reject_mixed_scopes_before_writing(tmp_path):
    import pytest
    scores = _scores()
    scores['cohort_scope'] = ['main_shared'] * 8 + ['icu_waveform_sensitivity']
    with pytest.raises(ValueError, match='cannot mix cohort_scope'):
        assets.export_core_comparisons(prepare_metrics_for_display(scores), tmp_path)
    assert not list(tmp_path.iterdir())
