"""Held-out calibration support from bundled predictions; no fitting or resampling."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '01_MANUSCRIPT/publication_tables'
DIAGNOSTICS = ROOT / '00_START_HERE/run_audit/calibration_overall_diagnostics.csv'
REGISTRY = ROOT / '04_DATA/prediction_index.csv'
CURVES = ROOT / '03_FIGURES/paper/S4-whole-case-calibration.plotted.csv'
OUTCOMES = {'any_aki': 'AKI', 'icu_admission': 'ICU admission'}
MODELS = {'logistic_regression': 'LR', 'xgboost': 'XGBoost', 'ebm': 'EBM'}
INPUTS = {'preop_only': 'Clinical only', 'preop_and_all_waveforms': 'Clinical + waveforms'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_support():
    """Reproduce the saved curve points and count predictions in fixed risk bins."""
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (REGISTRY, DIAGNOSTICS, CURVES)}
    index = pd.read_csv(REGISTRY)
    wanted = index[index.cohort_scope.eq('main_shared') & index.branch.eq('non_windowed') & index.feature_set.isin(INPUTS)].copy()
    assert len(wanted) == 12
    curves = pd.read_csv(CURVES)
    diagnostics = pd.read_csv(DIAGNOSTICS)
    records, checks = [], []
    identities = {}
    edges = np.linspace(0, 1, 41)
    for outcome in OUTCOMES:
        for model in MODELS:
            reference_identity = None
            for inputs in INPUTS:
                row = wanted[wanted.outcome.eq(outcome) & wanted.model.eq(model) & wanted.feature_set.eq(inputs)]
                assert len(row) == 1
                entry = row.iloc[0]
                path = ROOT / entry.offline_prediction_path
                source_hashes[str(path.relative_to(ROOT))] = sha(path)
                assert source_hashes[str(path.relative_to(ROOT))] == entry.pooled_prediction_sha256
                data = pd.read_csv(path)
                assert len(data) == 3571 and data.caseid.is_unique
                assert data.cohort_scope.eq('main_shared').all() and data.validation_scheme.eq('nested_cv').all()
                assert data.outcome.eq(outcome).all() and data.model_type.eq(model).all() and data.feature_set.eq(inputs).all()
                assert data.is_oof.all() and not data.diagnostic.any()
                assert data.y_prob_calibrated.between(0, 1).all() and data.y_true.isin([0, 1]).all()
                identity = data[['caseid', 'subjectid', 'y_true', 'outer_fold_id']].sort_values('caseid').reset_index(drop=True)
                if reference_identity is None:
                    reference_identity = identity
                else:
                    pd.testing.assert_frame_equal(identity, reference_identity)
                if outcome in identities:
                    pd.testing.assert_frame_equal(identity, identities[outcome])
                else:
                    identities[outcome] = identity
                risk = data.y_prob_calibrated
                groups = pd.qcut(risk, 10, duplicates='drop', labels=False)
                actual = data.assign(display_bin=groups).groupby('display_bin', observed=True).agg(
                    mean_risk=('y_prob_calibrated', 'mean'), observed_rate=('y_true', 'mean'),
                    n=('y_true', 'size'), events=('y_true', 'sum')).reset_index().rename(columns={'display_bin': 'bin'})
                saved = curves[curves.outcome.eq(outcome) & curves.family.eq(model) & curves.feature_set.eq(inputs)].sort_values('bin').reset_index(drop=True)
                assert len(saved) == len(actual) == 10
                errors = {}
                for column in ['mean_risk', 'observed_rate', 'n', 'events']:
                    error = float(np.max(np.abs(saved[column].to_numpy() - actual[column].to_numpy())))
                    assert error <= 1e-14, (outcome, model, inputs, column, error)
                    errors[column] = error
                diag = diagnostics[diagnostics.outcome.eq(outcome) & diagnostics.model.eq(model) & diagnostics.feature_set.eq(inputs)]
                assert len(diag) == 1
                diag = diag.iloc[0]
                values = {'operations': len(data), 'events': int(data.y_true.sum()),
                          'event_prevalence': float(data.y_true.mean()),
                          'mean_predicted_probability': float(risk.mean()),
                          'observed_expected_event_ratio': float(data.y_true.sum() / risk.sum()),
                          'brier_score': float(np.mean((risk - data.y_true) ** 2)),
                          'minimum_bin_n': int(actual.n.min()), 'maximum_bin_n': int(actual.n.max()),
                          'minimum_events_in_a_bin': int(actual.events.min())}
                for key, value in values.items():
                    assert np.isclose(value, diag[key], rtol=0, atol=1e-14), (key, value, diag[key])
                count, _ = np.histogram(risk, bins=edges)
                assert count.sum() == 3571
                for i, n in enumerate(count):
                    records.append({'outcome': outcome, 'family': model, 'feature_set': inputs,
                                    'bin': i, 'bin_lower': edges[i], 'bin_upper': edges[i + 1],
                                    'count': int(n), 'operations': len(data),
                                    'fraction': float(n / len(data))})
                checks.append({'outcome': outcome, 'model': model, 'feature_set': inputs,
                               'operations': len(data), 'patients': int(data.subjectid.nunique()),
                               'events': int(data.y_true.sum()), 'curve_bins_verified': 10,
                               'histogram_bins': len(count), 'curve_max_abs_errors': errors,
                               'histogram_count_sum': int(count.sum()), 'diagnostics_verified': list(values),
                               'pooled_prediction_sha256': source_hashes[str(path.relative_to(ROOT))]})
    support = pd.DataFrame(records)
    assert len(support) == 480
    assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items())
    return support, checks, source_hashes


def main():
    import build_publication_tables as publication
    support, checks, source_hashes = load_support()
    diagnostic_source = pd.read_csv(DIAGNOSTICS)
    order = pd.DataFrame([{'outcome': o, 'model': m, 'feature_set': f}
                          for o in OUTCOMES for m in MODELS for f in INPUTS])
    raw = order.merge(diagnostic_source, on=['outcome', 'model', 'feature_set'], validate='one_to_one')
    assert len(raw) == 12
    display = pd.DataFrame({'Outcome': raw.outcome.map(OUTCOMES), 'Model': raw.model.map(MODELS),
                            'Inputs': raw.feature_set.map(INPUTS),
                            'Observed risk (%)': raw.event_prevalence.map(lambda v: f'{100*v:.2f}'),
                            'Mean predicted risk (%)': raw.mean_predicted_probability.map(lambda v: f'{100*v:.2f}'),
                            'O/E': raw.observed_expected_event_ratio.map(lambda v: f'{v:.3f}'),
                            'Min. events/bin': raw.minimum_events_in_a_bin.astype(int)})
    notes = ('Pooled outer-test predictions from 3,571 operations (3,432 patients). '
             'O/E, observed events divided by expected events (sum of predicted probabilities); 1 denotes marginal agreement. '
             'Each of the ten quantile groups in Figure 3 contains 357 or 358 operations. '
             'AKI, acute kidney injury; ICU, intensive care unit; LR, logistic regression; EBM, explainable boosting machine.')
    assert raw.minimum_bin_n.eq(357).all() and raw.maximum_bin_n.eq(358).all()
    styles = {i: {column: {'top_rule': True} for column in display.columns}
              for i in raw.index if raw.loc[i, 'feature_set'] == 'preop_only'}
    sections = [{'heading': 'Whole-case models', 'dataframe': display, 'styles': styles}]
    publication.bundle('Table_S8_calibration_diagnostics', 'Table S8. Calibration diagnostics', raw, display,
                       sections, notes,
                       'Twelve clinical-only and combined whole-case models shown in Figure 3; descriptive diagnostics copied from the saved audit and independently reproduced from the bundled outer-test probabilities.',
                       list(source_hashes))
    record = publication.INDEX[-1]
    record['source_sha256'] = source_hashes
    (OUT / 'calibration_support_table_index.json').write_text(json.dumps({'tables': [record], 'source_sha256': source_hashes,
        'models_fitted': 0, 'new_uncertainty_intervals': False}, indent=2) + '\n')
    audit = {'status': 'passed', 'configurations': 12, 'saved_curve_points_verified': 120,
             'histogram_count_rows': len(support), 'histogram_bin_width': .025,
             'clinical_combined_case_patient_label_fold_identity': True, 'source_hashes_unchanged': True,
             'models_fitted': 0, 'calibrators_fitted': 0, 'bootstrap_draws': 0,
             'diagnostic_rows': len(raw), 'table_pdf_pages': record['pdf_pages'],
             'source_sha256': source_hashes, 'checks': checks}
    (ROOT / '00_START_HERE/qa/calibration_support_review.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ['status', 'configurations', 'saved_curve_points_verified', 'histogram_count_rows', 'table_pdf_pages']}, indent=2))


if __name__ == '__main__':
    main()
