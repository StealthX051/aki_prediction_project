"""Explain the waveform descriptors displayed in the EBM figures; no model fitting."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tarfile
import textwrap

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '01_MANUSCRIPT/publication_tables'
SOURCE = ROOT / '06_SOURCE_DOCUMENTS/waveform_descriptor_definitions/pycatch22-0.4.5.tar.gz'
FIGURES = {
    'Figure 4': ROOT / '03_FIGURES/narrative_revision/Figure4_EBM_global_importance.plotted.csv',
    'Figure 5': ROOT / '03_FIGURES/narrative_revision/Figure5_EBM_selected_functions.plotted.csv',
    'Figure S9': ROOT / '03_FIGURES/narrative_revision/S9_EBM_extended_functions.plotted.csv',
}
METHODS = ROOT / '05_CODE/project_source/METHODS.md'
REFERENCES = [
    {'reference': 'Lubba et al. (2019), catch22', 'url': 'https://doi.org/10.1007/s10618-019-00647-x'},
    {'reference': 'pycatch22 0.4.5 source release', 'url': 'https://pypi.org/project/pycatch22/0.4.5/'},
    {'reference': 'pycatch22 0.4.5 release record', 'url': 'https://github.com/DynamicsAndNeuralSystems/pycatch22/releases/tag/v0.4.5'},
]

# Definitions follow the release used for extraction, not later Catch22 revisions.
FEATURES = [
    dict(feature='Primus_AWP_SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1', channel='Airway pressure (AWP)',
         descriptor='Rescaled-range fluctuation',
         definition='Locates a crossover between two scaling regions in the log–log fluctuation curve, using detrended ranges of the cumulative signal.',
         reading='Fraction of tested scales before the crossover. Larger values place the crossover later in that sequence; this is not a scaling exponent.',
         unit='dimensionless fraction', filter='Low-pass 12 Hz', native_rate_hz=62.5,
         implementation='SC_FluctAnal.c',
         implementation_detail='SC_FluctAnal_2_50_1_logi_prop_r1 with lag=1 and how=rsrangefit; output=(firstMinInd+1)/ntt.'),
    dict(feature='Primus_CO2_SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', channel='Carbon dioxide (CO₂)',
         descriptor='Detrended fluctuation (50%)',
         definition='Locates a scaling crossover using root-mean-square detrended fluctuations of a cumulative signal formed from every second sample.',
         reading='Fraction of tested scales before the crossover. “50%” identifies the sampling step, not a CO₂ concentration or explained variance.',
         unit='dimensionless fraction', filter='Low-pass 8 Hz', native_rate_hz=62.5,
         implementation='SC_FluctAnal.c',
         implementation_detail='SC_FluctAnal_2_50_1_logi_prop_r1 with lag=2 and how=dfa; output=(firstMinInd+1)/ntt.'),
    dict(feature='SNUADC_ECG_II_CO_HistogramAMI_even_2_5', channel='Electrocardiogram (ECG)',
         descriptor='Auto mutual information',
         definition='Quantifies dependence between standardized signal values two samples apart, using five equally spaced amplitude bins.',
         reading='Natural-log information units (nats); larger values indicate stronger dependence at the 0.2-s lag.',
         unit='nats', filter='Band-pass 0.5–40 Hz', native_rate_hz=500,
         implementation='CO_AutoCorr.c',
         implementation_detail='CO_HistogramAMI_even_2_5; lag=2, five bins, natural logarithm. Requested ECG II key may contain loading-only V5 fallback.'),
    dict(feature='SNUADC_PLETH_DN_Mean', channel='Photoplethysmogram (PLETH)',
         descriptor='Processed mean',
         definition='Arithmetic mean of the filtered and resampled PLETH signal, computed without waveform standardization.',
         reading='Processed signal units; larger values indicate a higher average of the filtered trace, not a calibrated perfusion measure.',
         unit='processed PLETH signal units', filter='Band-pass 0.1–10 Hz', native_rate_hz=500,
         implementation='DN_Mean.c',
         implementation_detail='DN_Mean wrapper uses normalize=0; describes the band-pass filtered, resampled signal rather than the unprocessed waveform level.'),
    dict(feature='Primus_AWP_SB_BinaryStats_diff_longstretch0', channel='Airway pressure (AWP)',
         descriptor='Decreasing-run statistic',
         definition='Summarizes the longest run of negative successive differences, using the run-count convention in pycatch22 0.4.5.',
         reading='Run-count units; larger values indicate a longer decreasing run. The version-specific count is not reported as an exact duration.',
         unit='version-specific run-count units', filter='Low-pass 12 Hz', native_rate_hz=62.5,
         implementation='SB_BinaryStats.c',
         implementation_detail='Negative differences map to 0 and zero/positive differences to 1. The 0.4.5 boundary convention can differ by one from a direct count; later releases change this behavior.'),
    dict(feature='Primus_CO2_CO_trev_1_num', channel='Carbon dioxide (CO₂)',
         descriptor='Time reversibility',
         definition='Mean cubed one-sample change in the standardized signal: mean[(z(t+1) − z(t))³]. One sample spans 0.1\u00a0s.',
         reading='Dimensionless signed statistic. Positive and negative values distinguish opposite asymmetries in successive changes.',
         unit='dimensionless', filter='Low-pass 8 Hz', native_rate_hz=62.5,
         implementation='CO_AutoCorr.c',
         implementation_detail='CO_trev_1_num returns mean(pow(y[i+1]-y[i],3)) on the wrapper-standardized waveform.'),
]
NOTES = (
    'Whole-case features use operative signals resampled to 10 Hz. Before resampling: AWP, low-pass 12 Hz; CO₂, low-pass 8 Hz; '
    'ECG, band-pass 0.5–40 Hz; PLETH, band-pass 0.1–10 Hz. Filters use fourth-order Butterworth designs applied forward and backward. '
    'Catch22 descriptors use waveform z-scores; the Catch24 mean retains processed signal units. ECG II may use V5 when II is unavailable. '
    'Feature values form the horizontal axes in Figures 5 and S9; Figure 4 shows term importance in log-odds. '
    'Definitions: Lubba et al., Data Mining and Knowledge Discovery (2019), doi:10.1007/s10618-019-00647-x; pycatch22 0.4.5 source.'
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = list(FIGURES.values()) + [METHODS, SOURCE]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    with tarfile.open(SOURCE, 'r:gz') as archive:
        c_files = {name: archive.extractfile('pycatch22-0.4.5/src/C/' + name).read().decode()
                   for name in ['SC_FluctAnal.c', 'CO_AutoCorr.c', 'DN_Mean.c', 'SB_BinaryStats.c', 'catch22_wrap.c']}
    # Pin the implementation details that distinguish the paper's release from 1.x.
    assert 'diffTemp < 0 ? 0 : 1' in c_files['SB_BinaryStats.c']
    assert 'double stretch0 = i - last1' in c_files['SB_BinaryStats.c']
    assert 'return (firstMinInd+1)/ntt' in c_files['SC_FluctAnal.c']
    assert 'SC_FluctAnal_2_50_1_logi_prop_r1(y, size, 2, "dfa")' in c_files['SC_FluctAnal.c']
    assert 'python_wrapper_double(args, &DN_Mean, 0)' in c_files['catch22_wrap.c']
    assert 'python_wrapper_double(args, &CO_trev_1_num, 1)' in c_files['catch22_wrap.c']
    shown = {}
    for figure, path in FIGURES.items():
        features = pd.read_csv(path).feature.drop_duplicates()
        for feature in features[features.str.startswith(('Primus_', 'SNUADC_'))]:
            shown.setdefault(feature, []).append(figure)
    assert set(shown) == {row['feature'] for row in FEATURES}, (set(shown), FEATURES)
    raw = pd.DataFrame(FEATURES)
    raw['figures'] = raw.feature.map(lambda feature: '; '.join(shown[feature]))
    raw['extraction_version'] = 'pycatch22 0.4.5'
    raw['whole_case_rate_hz'] = 10
    raw['source_track'] = raw.feature.map(lambda value: next(
        track for key, track in [('Primus_AWP_', 'Primus/AWP'), ('Primus_CO2_', 'Primus/CO2'),
                                ('SNUADC_PLETH_', 'SNUADC/PLETH'), ('SNUADC_ECG_II_', 'SNUADC/ECG_II')]
        if value.startswith(key)))
    raw['standardization'] = raw.feature.map(lambda feature: 'processed signal; no z-scoring' if feature.endswith('DN_Mean') else 'waveform z-score within pycatch22')
    display = pd.DataFrame({
        'Descriptor / waveform': raw.apply(lambda row: row.descriptor + '\n' + row.channel + '\n' + row.figures.replace('Figure ', 'Fig. '), axis=1),
        'Definition': raw.definition,
        'Reading the feature value': raw.reading,
    })
    display = display.map(lambda value: '\n'.join(textwrap.fill(line, 52, break_long_words=False)
                                                for line in value.splitlines()))
    spec = importlib.util.spec_from_file_location('publication_tables', Path(__file__).with_name('build_publication_tables.py'))
    publication = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publication)
    publication.bundle('Table_S9_waveform_descriptors', 'Table S9. Waveform descriptors', raw, display,
                       [{'heading': 'Table S9. Waveform descriptors', 'dataframe': display}], NOTES,
                       'All six unique waveform descriptors displayed in Figures 4, 5 and S9; definitions checked against the extraction release source.',
                       list(hashes))
    entry = publication.INDEX[-1]
    entry['source_sha256'] = hashes
    entry['definition_references'] = REFERENCES
    entry['source_release_sha256'] = hashes[str(SOURCE.relative_to(ROOT))]
    assert entry['pdf_pages'] <= 2
    assert all(digest(ROOT / name) == value for name, value in hashes.items())
    result = {'tables': [entry], 'source_sha256': hashes, 'models_fitted': 0,
              'definition_version': 'pycatch22 0.4.5', 'source_hashes_unchanged': True,
              'figure_feature_coverage': shown, 'references': REFERENCES,
              'implementation_file_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in c_files.items()}}
    (OUT / 'waveform_descriptor_table_index.json').write_text(json.dumps(result, indent=2) + '\n')
    review = {'status': 'passed', 'waveform_features': len(raw), 'complete_displayed_feature_coverage': True,
              'pdf_pages': entry['pdf_pages'], 'source_files_unchanged': True,
              'pinned_release_checks': 6, 'source_sha256': hashes, 'models_fitted': 0,
              'reader_notes': 'Definitions and axis scales distinguish standardized Catch22 features from the processed Catch24 mean; no physiological mechanisms assigned.'}
    (ROOT / '02_TABLES/qa/waveform_descriptor_review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()
