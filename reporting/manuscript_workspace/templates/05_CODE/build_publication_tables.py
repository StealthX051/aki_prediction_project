"""Curate publication supplements from saved summaries; no model fitting or inference."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
from html import escape

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '02_TABLES'))
from rendering_support.table_assets import (  # noqa: E402
    _add_dataframe_to_doc, dataframe_to_html, write_markdown_sections,
    write_pdf_sections,
)

from publication_style import apply_document_style, WORD_TABLE_PT
OUT = ROOT / '01_MANUSCRIPT/publication_tables'
INPUT = ROOT / '02_TABLES/inputs'
SOURCE_HASHES = {}
INDEX = []
OUTCOMES = {'any_aki': 'AKI within 72 hours', 'icu_admission': 'ICU admission'}
FAMILIES = {'logistic_regression': 'LR', 'xgboost': 'XGBoost', 'ebm': 'EBM'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(path):
    SOURCE_HASHES[str(path.relative_to(ROOT))] = digest(path)
    return path


def read(name):
    return pd.read_csv(track(INPUT / name))


def pages(path):
    result = subprocess.run(['pdfinfo', str(path)], check=True, capture_output=True, text=True)
    return int(next(line.split(':', 1)[1] for line in result.stdout.splitlines() if line.startswith('Pages:')))


def bundle(stem, title, raw, display, sections, notes, selection, sources):
    """Use the existing A4 PDF renderer and native editable Word table helper."""
    raw.to_csv(OUT / (stem + '_raw.csv'), index=False)
    display.to_csv(OUT / (stem + '_display.csv'), index=False)
    # Notes once per standalone table; PDF sections carry the same operative key.
    word = apply_document_style(Document(), body_size=WORD_TABLE_PT)
    page = word.sections[0]
    page.orientation = WD_ORIENT.LANDSCAPE
    page.page_width, page.page_height = Inches(11.69), Inches(8.27)
    page.top_margin = page.bottom_margin = Inches(.45)
    page.left_margin = page.right_margin = Inches(.5)
    word.add_heading(title, 1)
    word.add_paragraph(notes)
    for section in sections:
        word.add_heading(section['heading'], 2)
        _add_dataframe_to_doc(word, section['dataframe'], section.get('styles'))
    word.save(OUT / (stem + '.docx'))
    md_sections = [{'heading': title, 'text': notes}] + sections
    write_markdown_sections(OUT / (stem + '.md'), title=title, sections=md_sections)
    write_pdf_sections(OUT / (stem + '.pdf'), title=title,
                       sections=[dict(section, notes=section.get('pdf_notes', notes)) for section in sections])
    fragments = []
    for section in sections:
        rendered = dataframe_to_html(section['dataframe'], title=section['heading'], styles=section.get('styles'))
        fragments.append(rendered[rendered.index('<h1>'):].replace('<h1>', '<h2>').replace('</h1>', '</h2>'))
    template = dataframe_to_html(display.head(0), title=title, notes=notes)
    prefix = template[:template.index('<h1>')]
    (OUT / (stem + '.html')).write_text(prefix + '<h1>' + escape(title) + '</h1><p>' + escape(notes) + '</p>' + ''.join(fragments))
    INDEX.append({'stem': stem, 'title': title, 'rows': len(raw), 'display_rows': len(display),
                  'pdf_pages': pages(OUT / (stem + '.pdf')), 'sources': sources, 'selection': selection,
                  'formats': ['raw.csv', 'display.csv', 'docx', 'pdf', 'html', 'md']})


def s1():
    name = 'Table_S1_descriptives_missingness.csv'
    source = read(name)
    raw = source[source.row_type.ne('feature_missingness')].copy()
    section_order = ['Demographics', 'Surgical characteristics', 'Clinical comorbidities and tests', 'Laboratory values']
    cohort_order = ['main_shared', 'icu_waveform_sensitivity']
    raw['_cohort_order'] = raw.cohort_scope.map(dict(zip(cohort_order, range(2))))
    raw['_section_order'] = raw.section.map(dict(zip(section_order, range(4))))
    assert raw[['_cohort_order', '_section_order']].notna().all().all()
    raw = raw.sort_values(['_cohort_order', '_section_order'], kind='stable').drop(columns=['_cohort_order', '_section_order']).reset_index(drop=True)
    assert len(raw) == 163 and source.row_type.eq('feature_missingness').sum() == 1112
    assert raw.branch.eq('non_windowed').all()
    def characteristic(row):
        if row.row_type == 'category_level':
            if row.variable == 'ASA' and str(row.category) in {'1.0', '2.0', '3.0', '4.0'}:
                return '    ' + str(int(float(row.category)))
            if row.variable == 'Inpatient prior to surgery' and str(row.category) == '1.0':
                return '    Yes'
            return '    ' + {'M': 'Male', 'F': 'Female', 'True': 'Yes', 'False': 'No'}.get(str(row.category), str(row.category))
        return row.variable.replace('10^3', '10³')
    alignment_keys = ['section', 'variable', 'category', 'row_type']
    aligned = None
    for cohort, prefix in [('main_shared', 'Main'), ('icu_waveform_sensitivity', 'Broader ICU')]:
        part = raw[raw.cohort_scope.eq(cohort)][alignment_keys + ['summary', 'missing_n_percent_display']].fillna('')
        part = part.rename(columns={'summary': prefix + ' value', 'missing_n_percent_display': prefix + ' missing n (%)'})
        aligned = part if aligned is None else aligned.merge(part, on=alignment_keys, how='outer', validate='one_to_one', sort=False)
    # Outer merge can reorder keys; restore the first appearance in the source.
    first = raw.fillna('').drop_duplicates(alignment_keys)[alignment_keys].reset_index(drop=True)
    first['_position'] = range(len(first))
    variable_order = {(section, variable): i for i, (section, variable) in enumerate(first[['section', 'variable']].drop_duplicates().itertuples(index=False, name=None))}
    first['_variable_order'] = [variable_order[(section, variable)] for section, variable in first[['section', 'variable']].itertuples(index=False, name=None)]
    aligned = aligned.merge(first, on=alignment_keys, validate='one_to_one').sort_values(['_variable_order', '_position']).reset_index(drop=True)
    display = pd.DataFrame({'Characteristic': aligned.apply(characteristic, axis=1)})
    for column in ['Main value', 'Main missing n (%)', 'Broader ICU value', 'Broader ICU missing n (%)']:
        display[column] = aligned[column].fillna('—')
    # Keep compact parentheses on the same line in the shared PDF exporter.
    display = display.map(lambda value: str(value).replace(' (', '\u00a0('))
    assert len(display) == 82 and len(raw) == 163
    sections = []
    # These explicit blocks retain complete categorical variables on each page.
    blocks = [
        ('Demographics and surgical characteristics', aligned.section.eq('Demographics') | aligned.variable.isin(['Emergency', 'Surgical department']),
         'Main cohort: 3,571 operations; broader ICU cohort: 5,765. Values are median (Q1, Q3) or n (%); percentages use all operations in the cohort. BMI, body mass index.'),
        ('Surgical characteristics (continued)', aligned.variable.isin(['Procedure type', 'Surgical approach', 'ASA', 'Anesthesia']),
         'Values are n (%). ASA, American Society of Anesthesiologists physical status. —, category absent.'),
        ('Surgical characteristics (continued); comorbidities and tests', aligned.variable.eq('Inpatient prior to surgery') | aligned.section.eq('Clinical comorbidities and tests'),
         'Values are n (%). ECG, electrocardiogram; PFT, pulmonary function test.'),
        ('Laboratory values', aligned.section.eq('Laboratory values'),
         'Values are median (Q1, Q3). Hb, hemoglobin; PT, prothrombin time; aPTT, activated partial thromboplastin time; AST/ALT, aspartate/alanine aminotransferase; BUN, blood urea nitrogen; WBC, white blood cell count; CRP, C-reactive protein; eGFR, estimated glomerular filtration rate.'),
    ]
    covered = []
    for heading, mask, key in blocks:
        indices = aligned.index[mask]
        covered.extend(indices)
        styles = {i: {'Characteristic': {'bold': True}} for i in indices if aligned.loc[i, 'row_type'] == 'categorical'}
        sections.append({'heading': heading, 'dataframe': display.loc[indices], 'styles': styles, 'pdf_notes': key})
    assert covered == list(aligned.index)
    bundle('Table_S1_clinical_descriptives', 'Table S1. Cohort characteristics and missingness', raw, display, sections,
           'Main cohort: 3,571 operations; broader ICU cohort: 5,765. Values are median (Q1, Q3) or n (%); percentages use all operations in the cohort. —, category absent. ASA, American Society of Anesthesiologists physical status; Hb, hemoglobin; BMI, body mass index; ECG, electrocardiogram; PFT, pulmonary function test; PT, prothrombin time; aPTT, activated partial thromboplastin time; AST/ALT, aspartate/alanine aminotransferase; BUN, blood urea nitrogen; WBC, white blood cell count; CRP, C-reactive protein; eGFR, estimated glomerular filtration rate.',
           'All 163 clinical descriptive records aligned side by side into 82 characteristics, in fixed clinical-section order; the broader cohort has one additional category. The 1,112 waveform-feature missingness records remain in the complete archive.', [name])


def s5():
    absolute_name = 'absolute_core_source_rows.csv'
    paired_name = 'Table_S5_paired_decision_curve_values.csv'
    absolute, paired = read(absolute_name), read(paired_name)
    rows = []
    for outcome, thresholds in [('any_aki', [.05, .10, .15, .20, .25, .30]), ('icu_admission', [.05, .10, .20, .30, .40, .50, .60])]:
        for model in FAMILIES:
            for threshold in thresholds:
                a = absolute[absolute.cohort_scope.eq('main_shared') & absolute.branch.eq('non_windowed') & absolute.outcome.eq(outcome) & absolute.model_type.eq(model) & np.isclose(absolute.threshold_probability, threshold)]
                clinical = a[a.feature_set.eq('preop_only')]
                combined = a[a.feature_set.eq('preop_and_all_waveforms')]
                p = paired[paired.cohort_scope.eq('main_shared') & paired.branch.eq('non_windowed') & paired.outcome.eq(outcome) & paired.reference_model.eq(model) & paired.candidate_model.eq(model) & paired.reference_feature_set.eq('preop_only') & paired.candidate_feature_set.eq('preop_and_all_waveforms') & np.isclose(paired.threshold_probability, threshold)]
                assert len(clinical) == len(combined) == len(p) == 1, (outcome, model, threshold)
                c, b, d = clinical.iloc[0], combined.iloc[0], p.iloc[0]
                assert np.isclose(c.treat_all, b.treat_all) and c.treat_none == b.treat_none == 0
                assert np.isclose(b.model - c.model, d.delta, rtol=1e-9, atol=1e-12)
                record = d.to_dict()
                record.update(clinical_net_benefit=c.model, combined_net_benefit=b.model, treat_all_net_benefit=b.treat_all, treat_none_net_benefit=0.)
                rows.append(record)
    raw = pd.DataFrame(rows)
    assert len(raw) == 39
    number = lambda value: f'{value * 100:.3f}'.replace('-', '−')
    display = pd.DataFrame({'Outcome': raw.outcome.map(OUTCOMES), 'Model': raw.candidate_model.map(FAMILIES),
                            'Threshold (%)': raw.threshold_probability.map(lambda value: f'{value*100:.0f}'),
                            'Clinical NB /100': raw.clinical_net_benefit.map(number),
                            'Combined NB /100': raw.combined_net_benefit.map(number),
                            'Treat-all NB /100': raw.treat_all_net_benefit.map(number),
                            'Δ NB /100 (95% CI)': raw.apply(lambda r: f'{number(r.delta)}\u00a0({number(r.lower)}–{number(r.upper)})', axis=1)})
    sections = []
    for outcome in OUTCOMES:
        idx = raw.index[raw.outcome.eq(outcome)]
        styles = {i: {column: {'top_rule': True} for column in display.columns} for i in idx if raw.loc[i, 'threshold_probability'] == .05}
        sections.append({'heading': OUTCOMES[outcome], 'dataframe': display.loc[idx].drop(columns='Outcome'), 'styles': styles})
    bundle('Table_S5_representative_decision_curves', 'Table S5. Net benefit at selected thresholds', raw, display, sections,
           'Whole-case main cohort. NB, net benefit per 100 operations; Δ = combined minus clinical-only. Parentheses show 95% pointwise paired patient-cluster bootstrap CIs for Δ. Treat-none NB = 0. LR, logistic regression; EBM, explainable boosting machine.',
           '39 saved-grid intersections: 6 AKI and 7 ICU thresholds × 3 same-family clinical-versus-combined comparisons. Values and intervals copied from saved summaries; no new bootstrap or model fitting.', [absolute_name, paired_name])


def s6():
    figure4 = pd.read_csv(track(ROOT / '03_FIGURES/paper/figure_4_ebm_response_functions.csv'))
    plotted = pd.read_csv(track(ROOT / '03_FIGURES/paper/S6-ebm-nonlinearity-missingness.plotted.csv'))
    # Fixed function order follows the revised main Fig. 5 then supplemental Fig. S9.
    terms = [('any_aki', 'preop_hb'), ('icu_admission', 'age'),
             ('icu_admission', 'Primus_CO2_CO_trev_1_num'),
             ('any_aki', 'Primus_AWP_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1'),
             ('any_aki', 'preop_wbc'), ('icu_admission', 'SNUADC_PLETH_DN_Mean')]
    assert len(terms) == len(set(terms)) == 6
    for outcome, feature in terms[:4]:
        assert ((figure4.outcome == outcome) & (figure4.feature == feature)).any()
    for panel in ['AKI · Observed WBC response', 'ICU · Processed pleth mean response']:
        assert plotted.panel.eq(panel).any()
    importance = read('Table_S6_EBM_fold_importance.csv')
    missing = read('Table_S6_selected_missing_bins.csv')
    importance = importance[importance.branch.eq('non_windowed') & importance.feature_set.eq('preop_and_all_waveforms')].copy()
    missing = missing[missing.branch.eq('non_windowed') & missing.feature_set.eq('preop_and_all_waveforms')].copy()
    importance['fold_number'] = importance.fold.str.removeprefix('outer_').astype(int)
    missing['fold_number'] = missing.fold.astype(str).str.removeprefix('outer_').astype(int)
    keys = ['outcome', 'branch', 'feature_set', 'feature', 'fold_number']
    wanted = pd.DataFrame([{'outcome': outcome, 'feature': feature, 'fold_number': fold, 'figure_term_order': order} for order, (outcome, feature) in enumerate(terms) for fold in range(5)])
    wanted['branch'], wanted['feature_set'] = 'non_windowed', 'preop_and_all_waveforms'
    raw = wanted.merge(importance, on=keys, how='left', validate='one_to_one').merge(missing.drop(columns='fold'), on=keys, how='left', validate='one_to_one')
    assert len(raw) == 30 and raw[['native_rank', 'heldout_rank', 'missing_train_n', 'missing_score', 'train_n']].notna().all().all()
    raw = raw.sort_values(['figure_term_order', 'fold_number'], kind='stable').reset_index(drop=True)
    labels = {'preop_hb': 'Hemoglobin', 'Primus_AWP_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1': 'AWP rescaled-range fluctuation', 'age': 'Age', 'Primus_CO2_CO_trev_1_num': 'CO₂ time reversibility', 'preop_wbc': 'WBC', 'SNUADC_PLETH_DN_Mean': 'Processed PLETH mean', 'preop_crp': 'CRP'}
    display = pd.DataFrame({'Outcome': raw.outcome.map(OUTCOMES), 'Term': raw.feature.map(labels), 'Fold': raw.fold_number + 1,
                            'Missing/total n': raw.apply(lambda r: f'{int(r.missing_train_n):,} / {int(r.train_n):,}', axis=1),
                            'Missing-bin score': raw.missing_score.map(lambda value: f'{value:.4f}'.replace('-', '−')),
                            'Native rank': raw.native_rank.astype(int), 'Held-out rank': raw.heldout_rank.astype(int)})
    sections = []
    # Preserve main and supplementary response-function order.
    for start, end, label in [(0, 3, 'Figure 5'), (3, 6, 'Figure S9')]:
        idx = raw.index[raw.figure_term_order.ge(start) & raw.figure_term_order.lt(end)]
        styles = {i: {column: {'top_rule': True} for column in display.columns} for i in idx if raw.loc[i, 'fold_number'] == 0}
        sections.append({'heading': label, 'dataframe': display.loc[idx], 'styles': styles})
    bundle('Table_S6_plotted_EBM_terms', 'Table S6. EBM term support and importance', raw, display, sections,
           'Whole-case combined EBM. Missing-bin scores are uncalibrated log-odds contributions. Counts are missing/total training operations. Native ranks use training-weighted importance; held-out ranks use mean absolute outer-test term contributions. AWP, airway pressure; WBC, white blood cell count; PLETH, photoplethysmogram.',
           'Six outcome–term pairs shown in Figure 5 and Figure S9 × all five outer folds; exact feature-ID joins validated one-to-one. Term order follows the figures.', ['Table_S6_EBM_fold_importance.csv', 'Table_S6_selected_missing_bins.csv', '03_FIGURES/paper/figure_4_ebm_response_functions.csv', '03_FIGURES/paper/S6-ebm-nonlinearity-missingness.plotted.csv'])


def copy_existing():
    names = [('Table_S2_main_all_configurations', 180), ('Table_S2B_main_paired_differences', 180), ('Table_S2_input_key', 15), ('Table_S3_broader_ICU_all_configurations', 90), ('Table_S3B_broader_ICU_paired_differences', 90), ('Table_S4_family_comparisons', 162)]
    for stem, count in names:
        source = ROOT / '02_TABLES/supplement'
        files = sorted(source.glob(stem + '.*')) + sorted(source.glob(stem + '_*.csv'))
        assert files and all((source / (stem + '.' + suffix)).exists() for suffix in ['docx', 'md', 'html'])
        for path in files:
            track(path)
            shutil.copy2(path, OUT / path.name)
            assert digest(path) == digest(OUT / path.name)
        display_path = OUT / (stem + ('_display.csv' if (OUT / (stem + '_display.csv')).exists() else '.csv'))
        assert len(pd.read_csv(display_path)) == count
        title = next(line.removeprefix('# ') for line in (OUT / (stem + '.md')).read_text().splitlines() if line.startswith('# '))
        if stem == 'Table_S4_family_comparisons' and not (source / (stem + '.pdf')).exists():
            raw_path = track(INPUT / (stem + '.csv'))
            shutil.copy2(raw_path, OUT / (stem + '_raw.csv'))
            assert digest(raw_path) == digest(OUT / (stem + '_raw.csv'))
            frame = pd.read_csv(display_path).fillna('')
            notes = 'Candidate-minus-reference differences (95% paired patient-cluster bootstrap CIs), matched by cohort, outcome, branch and input set. Negative Brier differences favor the candidate. AP, average precision; LR, logistic regression; EBM, explainable boosting machine; AUROC, area under the receiver operating characteristic curve.'
            sections = []
            for (cohort, outcome, branch), group in frame.groupby(['Cohort', 'Outcome', 'Branch'], sort=False):
                sections.append({'heading': f'{cohort} — {outcome} — {branch}',
                                 'dataframe': group.drop(columns=['Cohort', 'Outcome', 'Branch']), 'notes': notes})
            write_pdf_sections(OUT / (stem + '.pdf'), title=title, sections=sections)
        assert (OUT / (stem + '.pdf')).exists()
        INDEX.append({'stem': stem, 'title': title, 'rows': count, 'display_rows': count, 'pdf_pages': pages(OUT / (stem + '.pdf')), 'sources': [str(path.relative_to(ROOT)) for path in files], 'selection': 'Complete formatted table copied without alteration; no core-only metric extract.'})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s1()
    copy_existing()
    s5()
    s6()
    for name in ['cohort_context_table_index.json', 'calibration_support_table_index.json',
                 'waveform_descriptor_table_index.json']:
        extra = OUT / name
        if extra.exists():
            payload = json.loads(extra.read_text())
            SOURCE_HASHES.update(payload.get('source_sha256', {}))
            for item in payload['tables']:
                INDEX.append(item)
                SOURCE_HASHES.update(item.get('source_sha256', {}))
    assert all(digest(ROOT / name) == value for name, value in SOURCE_HASHES.items())
    payload = {'tables': INDEX, 'source_sha256': SOURCE_HASHES, 'source_hashes_unchanged': True,
               'models_fitted': 0, 'bootstrap_draws_generated': 0}
    (OUT / 'publication_table_index.json').write_text(json.dumps(payload, indent=2) + '\n')
    lines = ['# Publication supplementary tables', '', 'Each table has editable Word, PDF, HTML and Markdown copies. CSVs retain saved numeric precision; display CSVs reproduce the reading copies.', '', '| Table | Records | PDF pages | Selection |', '| --- | ---: | ---: | --- |']
    lines += [f"| {row['title']} | {row['rows']} | {row['pdf_pages']} | {row['selection']} |" for row in INDEX]
    lines += ['', 'Source files, selection rules and SHA-256 checks are recorded in publication_table_index.json. The complete electronic archive remains in 02_TABLES; this folder contains the curated publication set.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(INDEX, indent=2))


if __name__ == '__main__':
    main()
