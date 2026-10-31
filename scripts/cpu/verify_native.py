"""Bounded checks of actual native libraries outside the mocked pytest process."""
import importlib
from importlib.metadata import distributions, version
import json
import os
from pathlib import Path
import platform
import sys
from zipfile import ZipFile

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.datasets import make_classification
from packaging.utils import canonicalize_name
from model_creation.validation import build_model
from reporting.manuscript_assets import export_dataframe_bundle
from catch24_fixture import extract

out = Path(os.environ['VERIFY_OUTPUT'])
out.mkdir(parents=True, exist_ok=False)
report = {'python': platform.python_version(), 'executable': sys.executable, 'platform':platform.platform(), 'checks':{}, 'versions':{}}
assert platform.python_version() == '3.12.16', report['python']
installed = sorted(f"{canonicalize_name(d.metadata['Name'])}=={d.version}" for d in distributions())
(out/'installed-python.txt').write_text('\n'.join(installed)+'\n')
expected = (repo/'locks/installed-python-linux-64.txt').read_text().splitlines()
assert installed == expected, f'Installed distribution versions differ: {set(installed) ^ set(expected)}'
report['checks']['installed_python_versions'] = 'passed; 70 distributions including Conda-owned Python packages'
for dist in ['numpy','pandas','scipy','scikit-learn','matplotlib','optuna','xgboost-cpu','pytest','joblib','shap','vitaldb','pycatch22','interpret-core','numba','python-docx','PyYAML','plotly']:
    report['versions'][dist] = version(dist)

for module in ['data_preparation.step_01_cohort_construction','data_preparation.step_02_catch_22','data_preparation.step_03_preop_prep','data_preparation.step_04_intraop_prep','data_preparation.step_05_data_merge','model_creation.step_06_run_hpo','model_creation.step_07_train_evaluate','reporting.make_report','reporting.make_shap_figures','results_recreation.metrics_summary','run_catch22']:
    importlib.import_module(module)
report['checks']['active_module_imports'] = 'passed'

features = extract()
assert len(features['names']) == len(set(features['names'])) == 24
assert np.isfinite(features['values']).all()
reference = json.loads((repo/'tests/fixtures/catch24-python39-windows-reference.json').read_text())
assert features['names'] == reference['names'], 'Catch24 feature names/order changed'
np.testing.assert_allclose(features['values'], reference['values'], rtol=1e-10, atol=1e-10)
report['checks']['catch24_python39_windows_parity'] = {'status':'passed','rtol':1e-10,'atol':1e-10,'max_abs_delta':float(np.max(np.abs(np.array(features['values'])-reference['values']))),'scope':'one synthetic fixture, not clinical feature parity'}
(out/'catch24-python312.json').write_text(json.dumps(features,indent=2)+'\n')

X, y = make_classification(n_samples=160,n_features=6,n_informative=4,n_redundant=0,random_state=42)
X = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
xgb_model = build_model('xgboost', {'n_estimators':16,'max_depth':2},random_state=42,threads_per_model=2,scale_pos_weight=1.0)
xgb_model.fit(X.iloc[:120],y[:120])
prob = xgb_model.predict_proba(X.iloc[120:])[:,1]
assert prob.shape == (40,) and np.isfinite(prob).all() and np.logical_and(prob>=0,prob<=1).all()
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X.iloc[120:125])
assert np.asarray(shap_values).shape == (5,6) and np.isfinite(shap_values).all()
np.testing.assert_allclose(explainer.expected_value + np.asarray(shap_values).sum(axis=1),
                           xgb_model.predict(X.iloc[120:125], output_margin=True), rtol=1e-5, atol=1e-6)
xgb_model.save_model(out/'synthetic-xgboost.json')
loaded = xgb.XGBClassifier()
loaded.load_model(out/'synthetic-xgboost.json')
np.testing.assert_allclose(loaded.predict_proba(X.iloc[120:])[:,1],prob)
build = xgb.build_info()
assert not build.get('USE_CUDA', False), build
assert not build.get('USE_NCCL', False), build
report['checks']['xgboost_cpu_fit_predict_shap_roundtrip'] = {'status':'passed','build_info':build}

ebm = build_model('ebm', {'max_rounds':32,'smoothing_rounds':0,'outer_bags':2,'inner_bags':0},random_state=42,threads_per_model=2,scale_pos_weight=1.0)
ebm.fit(X.iloc[:120],y[:120])
prob = ebm.predict_proba(X.iloc[120:])[:,1]
assert prob.shape == (40,) and np.isfinite(prob).all() and np.logical_and(prob>=0,prob<=1).all()
global_explanation = ebm.explain_global()
explanation_data = global_explanation.data()
assert len(explanation_data['names']) == len(explanation_data['scores']) == X.shape[1]
assert np.isfinite(explanation_data['scores']).all()
global_explanation.visualize().write_html(out/'synthetic-ebm.html')
report['checks']['ebm_repository_factory_fit_predict_html'] = {'status':'passed','missing':ebm.missing,'max_bins':ebm.max_bins,'note':'bounded bags/rounds; production workload not benchmarked'}

table = pd.DataFrame({'check':['CPU environment'],'value':[1]})
export_dataframe_bundle(out/'synthetic-table',title='Synthetic environment verification',dataframe=table)
pd.testing.assert_frame_equal(pd.read_csv(out/'synthetic-table.csv'), table)
assert 'CPU environment' in (out/'synthetic-table.md').read_text()
with ZipFile(out/'synthetic-table.docx') as document:
    assert b'CPU environment' in document.read('word/document.xml')
pdf = (out/'synthetic-table.pdf').read_bytes()
assert pdf.startswith(b'%PDF-') and pdf.rstrip().endswith(b'%%EOF')
report['checks']['synthetic_table_csv_md_docx_pdf'] = 'passed; CSV roundtrip, Markdown/DOCX content and PDF structure; no visual review'
report['checks']['kaleido_browser_png'] = 'not installed/verified; optional EBM static export'
(out/'native-verification.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(report,indent=2,default=str))
