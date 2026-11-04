"""Inspect delivered native files independently of rendering style helpers."""
from pathlib import Path
import csv,hashlib,json,math,re,subprocess,zipfile,xml.etree.ElementTree as ET
import pandas as pd
from lxml import html
from docx import Document
P=Path(__file__).resolve().parents[1]
N={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
metrics=['AP','AUROC','Brier','Sensitivity','Specificity','Precision','F1','Accuracy']
sourcekeys=['auprc','auroc','brier','sensitivity','specificity','precision','f1','accuracy']
outcomes={'AKI':'any_aki','AKI within 72 hours':'any_aki','ICU admission':'icu_admission','In-hospital mortality':'mortality'}
branches={'Whole case':'non_windowed','Windowed':'windowed'}
models={'LR':'Logistic regression','Logistic regression':'Logistic regression','XGBoost':'XGBoost','EBM':'EBM'}
inputs=['preop_only','all_waveforms','preop_and_all_waveforms','ecg_only','pleth_only','co2_only','awp_only','preop_and_ecg','preop_and_pleth','preop_and_co2','preop_and_awp','preop_and_all_minus_ecg','preop_and_all_minus_pleth','preop_and_all_minus_co2','preop_and_all_minus_awp']
labels=['Clinical only','Waveforms only','Clinical + waveforms','ECG only','PLETH only','CO₂ only','AWP only','Clinical + ECG','Clinical + PLETH','Clinical + CO₂','Clinical + AWP','Clinical + waveforms except ECG','Clinical + waveforms except PLETH','Clinical + waveforms except CO₂','Clinical + waveforms except AWP']
reverse=dict(zip(labels,inputs))
def norm(value):return str(value).replace('\n',' ').strip()
def interval(row,key):return f'{row[key]:.3f} ({row[key+"_lower"]:.3f}–{row[key+"_upper"]:.3f})'.replace('-','−')
def cellbold(cell):
    runs=[r for r in cell._tc.iter(W+'r') if ''.join(r.itertext()).strip()]
    return bool(runs) and all((r.find(W+'rPr/'+W+'b') is not None and r.find(W+'rPr/'+W+'b').get(W+'val','1') not in {'0','false','off'}) for r in runs)
def expected_bold(df,columns):
    wanted=set();groups=[c for c in ('Outcome','Branch','Model') if c in df]
    for _,g in df.groupby(groups,sort=False,dropna=False):
        for c in columns:
            # Audit printed values, so equal visible values must receive equal emphasis.
            values={i:float(str(df.at[i,c]).split()[0].replace('−','-')) for i in g.index}
            best=(min if c=='Brier' else max)(values.values())
            wanted.update((i,df.columns.get_loc(c)) for i,v in values.items() if v==best)
    return wanted
checks={'source_csv_hashes':0,'complete_metric_interval_cells':0,'main_metric_interval_cells':0,'word_metric_cells_checked':0,'bold_metric_cells':{},'displayed_tie_groups':0}
manifest=json.loads((P/'inputs/manifest.json').read_text())
for entry in manifest['sources']:
    assert hashlib.sha256((P/entry['file']).read_bytes()).hexdigest()==entry['sha256'],entry['file']
    checks['source_csv_hashes']+=1
bundles=[('supplement/Table_S2_core_all_metrics','main_shared',False),('supplement/Table_S2_main_all_configurations','main_shared',False),('supplement/Table_S2B_main_paired_differences','main_shared',True),('supplement/Table_S3_broader_ICU_all_configurations','icu_waveform_sensitivity',False),('supplement/Table_S3B_broader_ICU_paired_differences','icu_waveform_sensitivity',True),('meeting_backup/Meeting_backup_mortality_all_metrics','mortality_waveform_sensitivity',False)]
for rel,scope,delta in bundles:
    raw=pd.read_csv(P/'inputs'/f'source_{scope}_metrics_summary.csv').set_index(['outcome','branch','model_name','feature_set'])
    d=pd.read_csv(P/(rel+'_display.csv'),keep_default_na=False)
    for _,r in d.iterrows():
        row=raw.loc[(outcomes[r.Outcome],branches[r.Branch],models[r.Model],reverse[r['Feature set']])]
        for key,label in zip(sourcekeys,metrics):
            actual=r[('Δ ' if delta else '')+label]
            expected='Reference' if delta and r['Feature set']=='Clinical only' else interval(row,('delta_' if delta else '')+key)
            assert actual==expected,(rel,r['Feature set'],label,actual,expected)
            checks['complete_metric_interval_cells']+=1
    # One native table per endpoint/branch/family; independently verify all actual cells.
    doc=Document(P/(rel+'.docx'));flat=[]
    for table in doc.tables:
        assert all(cellbold(c) for c in table.rows[0].cells),(rel,'header bold')
        flat.extend(table.rows[1:])
    assert len(flat)==len(d),(rel,len(flat),len(d))
    for row,r in zip(flat,d.itertuples(index=False,name=None)):
        assert [norm(c.text) for c in row.cells]==[norm(v) for v in r[3:]],rel
    if not delta:
        wanted=expected_bold(d,metrics);actual=set()
        for i,row in enumerate(flat):
            for j,c in enumerate(metrics,1):
                coordinate=(i,d.columns.get_loc(c))
                if cellbold(row.cells[j]):actual.add(coordinate)
                checks['word_metric_cells_checked']+=1
        assert actual==wanted,(rel,'native metric bold discrepancy',actual^wanted)
        checks['bold_metric_cells'][rel]=len(actual)
        for _,g in d.groupby(['Outcome','Branch','Model'],sort=False):
            for c in metrics:
                vals=g[c].map(lambda x:float(x.split()[0].replace('−','-')))
                best=vals.min() if c=='Brier' else vals.max()
                if sum(vals==best)>1:checks['displayed_tie_groups']+=1
        text=(P/(rel+'.md')).read_text();h=(P/(rel+'.html')).read_text()
        assert '****' not in text
        # Every table-cell occurrence must carry the same expected emphasis in Markdown/HTML.
        mdrows=[line for line in text.splitlines() if line.startswith('| ') and not line.startswith('| Feature') and not re.match(r'^\| [-: ]+\|',line)]
        assert len(mdrows)==len(d),(rel,len(mdrows),len(d))
        for i,line in enumerate(mdrows):
            cells=[x.strip() for x in line.strip('|').split('|')]
            for j,c in enumerate(metrics,1):assert cells[j].startswith('**')==((i,d.columns.get_loc(c)) in wanted),(rel,i,c)
        tree=html.fromstring(h);blocks=re.findall(r'([^{}]+)\{([^{}]+)\}',h)
        boldids=set()
        for selectors,body in blocks:
            if re.search(r'font-weight:\s*bold',body):boldids.update(re.findall(r'#([\w-]+_row\d+_col\d+)',selectors))
        hrows=tree.xpath('//tbody/tr');assert len(hrows)==len(d)
        for i,row in enumerate(hrows):
            for j,c in enumerate(metrics,1):
                cell=row.xpath('./td')[j]
                assert (cell.get('id') in boldids)==((i,d.columns.get_loc(c)) in wanted),(rel,i,c,'HTML')
mainrel='main/Table_2_core_comparisons';d=pd.read_csv(P/(mainrel+'.csv'),keep_default_na=False)
raw=pd.read_csv(P/'inputs/source_main_shared_metrics_summary.csv').set_index(['outcome','branch','model_name','feature_set'])
for _,r in d.iterrows():
    sr=raw.loc[(outcomes[r.Outcome],'non_windowed',models[r.Model],reverse[r['Feature set']])]
    for key,c in [('auprc','AP'),('auroc','AUROC'),('brier','Brier'),('delta_auprc','Δ AP vs clinical')]:
        assert r[c]==('Reference' if key=='delta_auprc' and r['Feature set']=='Clinical only' else interval(sr,key))
        checks['main_metric_interval_cells']+=1
wanted=expected_bold(d,metrics[:3]);table=Document(P/(mainrel+'.docx')).tables[0];assert len(table.rows)==19
assert all(cellbold(c) for c in table.rows[0].cells)
actual={(i,j) for i,row in enumerate(table.rows[1:]) for j in range(3,6) if cellbold(row.cells[j])}
assert wanted==actual,('main bold',actual^wanted)
# Regression: both EBM AKI Brier values visibly round to .040 and must both remain bold.
assert (6,5) in actual and (8,5) in actual and d.iloc[6].Brier.split()[0]==d.iloc[8].Brier.split()[0]=='0.040'
assert all(cellbold(table.rows[i+1].cells[1]) for i in range(0,18,3))
assert all(cellbold(table.rows[i+1].cells[0]) for i in (0,9))
checks['bold_metric_cells'][mainrel]=len(actual);checks['word_metric_cells_checked']+=54
checks['main_table_2_metric_bold_coordinates_zero_based']=[list(c) for c in sorted(actual)]
checks['main_table_2_structure']={'bold_header_cells':7,'bold_family_labels':6,'bold_outcome_labels':2}
mdrows=[l for l in (P/(mainrel+'.md')).read_text().splitlines() if l.startswith('|')][2:]
assert len(mdrows)==18
for i,line in enumerate(mdrows):
    cells=[x.strip() for x in line.strip('|').split('|')]
    for j in range(3,6):assert cells[j].startswith('**')==((i,j) in wanted)
assert '****' not in (P/(mainrel+'.md')).read_text()
h=(P/(mainrel+'.html')).read_text();boldids=set()
for selectors,body in re.findall(r'([^{}]+)\{([^{}]+)\}',h):
    if re.search(r'font-weight:\s*bold',body):boldids.update(re.findall(r'#([\w-]+_row\d+_col\d+)',selectors))
for i,row in enumerate(html.fromstring(h).xpath('//tbody/tr')):
    for j in range(3,6):assert (row.xpath('./td')[j].get('id') in boldids)==((i,j) in wanted)
# Full XML spreadsheet round-trip: typed sources remain unchanged; reading cells retain text.
wm=json.loads((P/'qa/workbook_manifest.json').read_text())
sourcefiles=['source_main_shared_metrics_summary.csv','source_icu_waveform_sensitivity_metrics_summary.csv','source_mortality_waveform_sensitivity_metrics_summary.csv','Table_1_cohort.csv','Table_S1_descriptives_missingness.csv','Table_S4_family_comparisons.csv','Table_S5_paired_decision_curve_values.csv','Table_S6_EBM_fold_importance.csv','Table_S6_selected_missing_bins.csv']
sourcefiles += [f'{scope}__decision_curve_{kind}.csv' for scope in ['main_shared','icu_waveform_sensitivity','mortality_waveform_sensitivity'] for kind in ['summary','comparisons']]
checks['spreadsheet_cells_round_tripped']=0;checks['spreadsheet_sheets']={};checks['excel_main_metric_bold_cells']=0
for filename,items,rawmode in [('Manuscript_Tables_Editable.xlsx',wm['reading_sheets'],False),('Complete_Numeric_Sources.xlsx',[{'file':'inputs/'+s} for s in sourcefiles],True)]:
    with zipfile.ZipFile(P/'workbooks'/filename) as z:
        workbook=ET.fromstring(z.read('xl/workbook.xml'));sheets=workbook.findall('s:sheets/s:sheet',N);assert len(sheets)==16
        checks['spreadsheet_sheets'][filename]=len(sheets)
        styles=ET.fromstring(z.read('xl/styles.xml')).findall('s:cellXfs/s:xf',N)
        for sheetno,item in enumerate(items,2):
            df=pd.read_csv(P/item['file'],keep_default_na=rawmode);root=ET.fromstring(z.read(f'xl/worksheets/sheet{sheetno}.xml'))
            rows=root.findall('s:sheetData/s:row',N)[3:];assert len(rows)==len(df),(filename,item,len(rows),len(df))
            for i,(xr,row) in enumerate(zip(rows,df.itertuples(index=False,name=None))):
                cells=xr.findall('s:c',N);assert len(cells)==len(row)
                for j,(xc,value) in enumerate(zip(cells,row)):
                    t=xc.get('t');child=xc.find('s:is/s:t',N) if t=='inlineStr' else xc.find('s:v',N)
                    actualvalue=None if child is None else child.text or ''
                    if pd.isna(value):assert actualvalue is None,(filename,i,j)
                    elif t=='inlineStr':
                        expected=str(value) if rawmode else str(value).replace(' (','\n(')
                        assert actualvalue==expected,(filename,item,i,j,actualvalue,expected)
                    else:assert float(actualvalue)==float(value),(filename,item,i,j)
                    if filename=='Manuscript_Tables_Editable.xlsx' and item['worksheet']=='Table2_Main' and 3<=j<=5:
                        bold=styles[int(xc.get('s'))].get('fontId')=='1'
                        assert bold==((i,j) in wanted),('Excel main bold',i,j)
                        checks['excel_main_metric_bold_cells']+=int(bold)
                    checks['spreadsheet_cells_round_tripped']+=1
# Verify all electronic reading rows remain present, with correctly attached numeric values.
from collections import Counter
flabels=json.loads((P/'inputs/feature_labels.json').read_text())
inputnames=dict(zip(inputs,labels));outnames={'any_aki':'AKI within 72 hours','icu_admission':'ICU admission','mortality':'In-hospital mortality'}
branchnames={'non_windowed':'Whole case','windowed':'Windowed'};modelnames={'logistic_regression':'LR','xgboost':'XGBoost','ebm':'EBM'}
cohortnames={'main_shared':'Main shared cohort','icu_waveform_sensitivity':'Broader ICU cohort'}
num=lambda v,n=3:f'{v:.{n}f}'
span=lambda a,b,c:f'{a:.3f} ({b:.3f}–{c:.3f})'.replace('-','−')
checks['electronic_reading_rows_verified']={}
for source,display,kind in [('Table_S4_family_comparisons.csv','Table_S4_family_comparisons_display.csv','family'),('Table_S5_paired_decision_curve_values.csv','Table_S5_paired_decision_curve_values_display.csv','paired'),('Table_S6_EBM_fold_importance.csv','Table_S6_EBM_fold_importance_display.csv','terms'),('Table_S6_selected_missing_bins.csv','Table_S6B_selected_missing_bins_display.csv','bins')]:
    raw=pd.read_csv(P/'inputs'/source);shown=pd.read_csv(P/'supplement'/display,keep_default_na=False);expected=[]
    if kind=='family':
        for _,r in raw.iterrows():expected.append((cohortnames[r.cohort_scope],outnames[r.outcome],branchnames[r.branch],inputnames[r.reference_feature_set],modelnames[r.reference_model],modelnames[r.candidate_model],{'auprc':'AP','auroc':'AUROC','brier':'Brier'}[r.metric],num(r.reference_estimate),num(r.candidate_estimate),span(r.delta,r.lower,r.upper)))
    elif kind=='paired':
        for _,r in raw.iterrows():expected.append((cohortnames[r.cohort_scope],outnames[r.outcome],branchnames[r.branch],modelnames[r.reference_model],modelnames[r.candidate_model],inputnames[r.reference_feature_set],inputnames[r.candidate_feature_set],num(r.threshold_probability*100,1),span(r.delta*100,r.lower*100,r.upper*100)))
    elif kind=='terms':
        for _,r in raw.iterrows():
            d=r.drop(labels=['family','group']).to_dict();d['outcome']=outnames[r.outcome];d['branch']=branchnames[r.branch];d['feature_set']=inputnames[r.feature_set];d['fold']='Fold '+str(int(r.fold.removeprefix('outer_'))+1);d['feature']=flabels.get(r.feature,r.feature).replace('Pleth —','PLETH —')
            for c in ('native_rank','heldout_rank'):d[c]=str(int(r[c]))
            for c in ('native_importance','heldout_mean_absolute'):d[c]=num(r[c],6)
            expected.append(tuple(str(d[c]) for c in raw.columns if c not in {'family','group'}))
    else:
        for _,r in raw.iterrows():expected.append((outnames[r.outcome],branchnames[r.branch],inputnames[r.feature_set],'Fold '+str(int(r.fold.removeprefix('outer_'))+1),r.label.replace('Pleth —','PLETH —'),num(r.missing_score,4),str(int(r.missing_train_n)),str(int(r.train_n)),str(int(r.missing_test_n)),str(int(r.test_n)),num(r.calibration_slope,4)))
    # Read all textual fields as supplied rather than inferring display precision as floats.
    shown=pd.read_csv(P/'supplement'/display,dtype=str,keep_default_na=False)
    actual=[tuple(r) for r in shown.itertuples(index=False,name=None)]
    assert Counter(actual)==Counter(expected),(source,'electronic numeric/identity mismatch',list((Counter(actual)-Counter(expected)).items())[:1])
    checks['electronic_reading_rows_verified'][source]=len(raw)

# Check every delivered editable Word table still includes its entire reading dataset.
checks['native_word_files']={}
for f in sorted([*P.glob('main/*.docx'),*P.glob('supplement/*.docx'),*P.glob('meeting_backup/*.docx')]):
    doc=Document(f);n=sum(len(t.rows)-1 for t in doc.tables)
    checks['native_word_files'][str(f.relative_to(P))]={'tables':len(doc.tables),'rows':n}
assert len(checks['native_word_files'])==15
assert checks['native_word_files']['supplement/Table_S6_EBM_fold_importance.docx']['rows']==11294
# PDF proof audit: visible bold points survive export and all words remain on paper.
checks['pdf_pages']={};checks['off_page_pdf_words']=0
for f in sorted([*P.glob('main/*.pdf'),*P.glob('supplement/*.pdf'),*P.glob('meeting_backup/*.pdf')]):
    root=ET.fromstring(subprocess.check_output(['pdftotext','-bbox',str(f),'-']))
    pages=list(root.iter('{http://www.w3.org/1999/xhtml}page'));checks['pdf_pages'][str(f.relative_to(P))]=len(pages)
    for page in pages:
        width=float(page.get('width'));height=float(page.get('height'))
        for word in page.iter('{http://www.w3.org/1999/xhtml}word'):
            if float(word.get('xMin'))<0 or float(word.get('yMin'))<0 or float(word.get('xMax'))>width or float(word.get('yMax'))>height:checks['off_page_pdf_words']+=1
    if f.parent.name=='main':assert len(pages)==1
assert checks['off_page_pdf_words']==0
pdfxml=ET.fromstring(subprocess.check_output(['pdftohtml','-xml','-stdout',str(P/(mainrel+'.pdf'))]))
pdfpoints=[e for e in pdfxml.iter('text') if re.fullmatch(r'0\.\d{3}',''.join(e.itertext()).strip()) and e.find('b') is not None]
assert len(pdfpoints)==len(wanted),('PDF bold points',len(pdfpoints),len(wanted))
checks['pdf_main_bold_metric_points']=len(pdfpoints)
phrases=['rather than causal','not causal','not confidence interval','not a confidence interval','supplied separately','fold-to-fold spread','fitted association rather than']
for f in [*P.glob('main/*.md'),*P.glob('supplement/*.md'),*P.glob('meeting_backup/*.md')]:
    lower=f.read_text().lower();assert not any(s in lower for s in phrases),(f,'repeated policy prose')
s4source=P/'inputs/Table_S4_family_comparisons.csv';s4raw=P/'supplement/Table_S4_family_comparisons_raw.csv'
assert s4raw.read_bytes()==s4source.read_bytes()
s4=pd.read_csv(s4source);s4expected=[f'{getattr(row,key):.3f}'.replace('-','−') for row in s4.itertuples() for key in ('reference_estimate','candidate_estimate','delta','lower','upper')]
s4text=subprocess.check_output(['pdftotext','-layout',str(P/'supplement/Table_S4_family_comparisons.pdf'),'-'],text=True)
s4actual=re.findall(r'(?<!\d)[−-]?\d\.\d{3}(?!\d)',s4text)
assert Counter(s4actual)==Counter(s4expected)
checks['s4_complete_bundle']={'raw_csv_exact_source_copy':True,'source_rows':len(s4),'pdf_numeric_values_verified':len(s4actual),'a4_landscape_pages':checks['pdf_pages']['supplement/Table_S4_family_comparisons.pdf'],'off_page_words':0}
checks['footnote_review']='Essential summary/CI, population, units, reference and bold key retained; repeated provenance, causal and uncertainty-policy prose removed.'
checks['displayed_precision_tie_regression']='Both AKI/EBM Brier point estimates displayed as 0.040 are bold in native Word, Markdown, HTML, PDF and reading Excel.'
checks['status']='passed';checks['reviewed_date']='2026-11-03';checks['desktop_office_limitation']='Native OOXML and PDF proofs inspected; Microsoft Word/LibreOffice are unavailable on this VM.'
(P/'qa/publication_editorial_review.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
