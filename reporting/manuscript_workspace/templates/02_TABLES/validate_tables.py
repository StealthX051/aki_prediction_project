"""Independent data/format checks for the portable manuscript tables."""
from pathlib import Path
import hashlib,json,re,subprocess,zipfile,xml.etree.ElementTree as ET
import pandas as pd
HERE=Path(__file__).resolve().parent
METRICS={'AP':'auprc','AUROC':'auroc','Brier':'brier','Sensitivity':'sensitivity','Specificity':'specificity','Precision':'precision','F1':'f1','Accuracy':'accuracy'}
OUTCOMES={'AKI within 72 hours':'any_aki','ICU admission':'icu_admission','In-hospital mortality':'mortality'}
BRANCHES={'Whole case':'non_windowed','Windowed':'windowed'}
FEATURES={'Clinical only':'preop_only','Waveforms only':'all_waveforms','Clinical + waveforms':'preop_and_all_waveforms'}
for c,label in [('ecg','ECG'),('pleth','PLETH'),('co2','CO₂'),('awp','AWP')]:
    FEATURES[label+' only']=c+'_only';FEATURES['Clinical + '+label]='preop_and_'+c;FEATURES['Clinical + waveforms except '+label]='preop_and_all_minus_'+c


def fmt(row,key):return f'{row[key]:.3f} ({row[key+"_lower"]:.3f}–{row[key+"_upper"]:.3f})'.replace('-','−')

def validate(base=HERE):
 base=Path(base);checks={};sources=json.loads((base/'inputs/manifest.json').read_text())
 for item in sources['sources']:
    p=base/item['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];assert len(pd.read_csv(p))==item['rows']
 checks['unchanged_source_csvs']=len(sources['sources'])
 count=0
 books=[('Table_S2_main_all_configurations',180,False),('Table_S2_core_all_metrics',36,False),('Table_S2B_main_paired_differences',180,True),('Table_S3_broader_ICU_all_configurations',90,False),('Table_S3B_broader_ICU_paired_differences',90,True),('Meeting_backup_mortality_all_metrics',18,False)]
 for stem,n,delta in books:
    folder='meeting_backup' if stem.startswith('Meeting') else 'supplement'
    raw=pd.read_csv(base/folder/(stem+'_raw.csv'));display=pd.read_csv(base/folder/(stem+'_display.csv'));assert len(raw)==len(display)==n
    source_name='source_mortality_waveform_sensitivity_metrics_summary.csv' if stem.startswith('Meeting') else 'source_icu_waveform_sensitivity_metrics_summary.csv' if 'S3' in stem else 'source_main_shared_metrics_summary.csv'
    source=pd.read_csv(base/'inputs'/source_name)
    ids=['outcome','branch','model_name','feature_set'];index=source.set_index(ids)
    for row in display.to_dict('records'):
        key=(OUTCOMES[row['Outcome']],BRANCHES[row['Branch']],row['Model'],FEATURES[row['Feature set']])
        ref=index.loc[key]
        for label,metric in METRICS.items():
            expected='Reference' if delta and key[3]=='preop_only' else fmt(ref,('delta_' if delta else '')+metric)
            assert row[('Δ ' if delta else '')+label]==expected,(stem,key,label,row)
            count+=1
    assert raw.set_index(ids).index.is_unique
 checks['metric_interval_display_cells']=count;assert count==4752
 d=pd.read_csv(base/'main/Table_2_core_comparisons.csv');raw=pd.read_csv(base/'inputs/source_main_shared_metrics_summary.csv').set_index(['outcome','branch','model_name','feature_set'])
 for row in d.to_dict('records'):
    key=('any_aki' if row['Outcome']=='AKI' else 'icu_admission','non_windowed','Logistic regression' if row['Model']=='LR' else row['Model'],FEATURES[row['Feature set']]);ref=raw.loc[key]
    for label,keyname in [('AP','auprc'),('AUROC','auroc'),('Brier','brier'),('Δ AP vs clinical','delta_auprc')]:
        assert row[label]==('Reference' if keyname=='delta_auprc' and key[3]=='preop_only' else fmt(ref,keyname))
 checks['main_compact_metric_cells']=72
 absraw=pd.read_csv(base/'inputs/absolute_core_source_rows.csv');absdisplay=pd.read_csv(base/'supplement/Table_S5A_absolute_net_benefit_display.csv')
 index=absraw.set_index(['outcome','model_name','feature_set','threshold_probability'])
 for row in absdisplay.to_dict('records'):
    key=(OUTCOMES[row['Outcome']],'Logistic regression' if row['Model']=='LR' else row['Model'],FEATURES[row['Feature set']],row['Threshold (%)']/100)
    # Decimal threshold representation may differ by an insignificant parser ulp.
    choices=index.loc[key[:3]];ref=choices.iloc[(abs(choices.index-key[3])).argmin()]
    assert abs(float(ref.name)-key[3])<1e-12
    for source,col in [('model','Model net benefit /100 (95% CI)'),('treat_all','Treat-all net benefit /100 (95% CI)')]:
        expected=f'{ref[source]*100:.3f} ({ref[source+"_lower"]*100:.3f}–{ref[source+"_upper"]*100:.3f})'.replace('-','−');assert row[col]==expected
    assert row['Treat-none net benefit /100']==0
 checks['conventional_dca_reading_rows']=len(absdisplay);assert len(absdisplay)==774
 wm=json.loads((base/'qa/workbook_manifest.json').read_text());total_cells=0;workbook_sheets={}
 raw_sources={'Main_180':'source_main_shared_metrics_summary.csv','Broader_ICU_90':'source_icu_waveform_sensitivity_metrics_summary.csv','Mortality_18':'source_mortality_waveform_sensitivity_metrics_summary.csv','Cohort_Source':'Table_1_cohort.csv','S1_Complete':'Table_S1_descriptives_missingness.csv','S4_Comparisons':'Table_S4_family_comparisons.csv','S5_Paired_DCA':'Table_S5_paired_decision_curve_values.csv','S6_EBM_Terms':'Table_S6_EBM_fold_importance.csv','S6_Missing_Bins':'Table_S6_selected_missing_bins.csv'}
 for scope,short in [('main_shared','Main'),('icu_waveform_sensitivity','ICU'),('mortality_waveform_sensitivity','Mortality')]:
    for kind,label in [('summary','Absolute'),('comparisons','Paired')]:raw_sources[f'DCA_{short}_{label}']=f'{scope}__decision_curve_{kind}.csv'
 ns={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 for filename,reading in [('Manuscript_Tables_Editable.xlsx',True),('Complete_Numeric_Sources.xlsx',False)]:
    with zipfile.ZipFile(base/'workbooks'/filename) as z:
        assert z.testzip() is None
        for name in z.namelist():ET.fromstring(z.read(name))
        wb=ET.fromstring(z.read('xl/workbook.xml'));sheets=wb.find('x:sheets',ns);workbook_sheets[filename]=len(sheets)
        assert not any('externalLinks' in name for name in z.namelist())
        for i,sheet in enumerate(sheets,1):
            name=sheet.attrib['name'];root=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
            assert root.find('.//x:pane',ns).attrib['topLeftCell']=='A5'
            assert root.find('x:autoFilter',ns) is not None
            if name=='READ_ME':continue
            if reading:
                item=next(item for item in wm['reading_sheets'] if item['worksheet']==name);frame=pd.read_csv(base/item['file'],keep_default_na=False)
            else:frame=pd.read_csv(base/'inputs'/raw_sources[name])
            rows=[row for row in root.find('x:sheetData',ns) if int(row.attrib['r'])>=5];assert len(rows)==len(frame)
            for r,values in zip(rows,frame.itertuples(index=False,name=None)):
                assert len(r)==len(values)
                for cell,expected in zip(r,values):
                    if pd.isna(expected):assert cell.find('x:v',ns) is None and cell.find('x:is',ns) is None;continue
                    if cell.attrib.get('t')=='inlineStr':
                        actual=''.join(cell.find('x:is',ns).itertext());want=str(expected).replace(' (','\n(') if reading and isinstance(expected,str) else str(expected);assert actual==want,(filename,name,actual,want)
                    elif cell.attrib.get('t')=='b':assert bool(int(cell.find('x:v',ns).text))==bool(expected)
                    else:assert float(cell.find('x:v',ns).text)==float(expected),(filename,name,cell.attrib,expected)
                    total_cells+=1
 checks['spreadsheet_sheets']=workbook_sheets;checks['spreadsheet_data_cells_round_tripped']=total_cells
 word={};wns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
 for p in sorted(base.glob('*/*.docx')):
    with zipfile.ZipFile(p) as z:
        root=ET.fromstring(z.read('word/document.xml'));tables=root.findall('.//w:tbl',wns);assert tables
        rows=[r for t in tables for r in t.findall('w:tr',wns)];headers=[t.find('w:tr/w:trPr/w:tblHeader',wns) for t in tables];assert all(h is not None for h in headers)
        assert all(r.find('w:trPr/w:cantSplit',wns) is not None for r in rows)
        assert not root.findall('.//w:drawing',wns)
        word[str(p.relative_to(base))]={'native_tables':len(tables),'data_rows':len(rows)-len(tables)}
 checks['native_editable_word_files']=word;assert len(word)==15
 assert word['supplement/Table_S6_EBM_fold_importance.docx']['data_rows']==11294
 pdf_checks={};off=[]
 for p in sorted(base.glob('*/*.pdf')):
    result=subprocess.run(['pdftotext','-bbox',str(p),'-'],capture_output=True,text=True,check=True);root=ET.fromstring(result.stdout);pages=list(root.iter('{http://www.w3.org/1999/xhtml}page'))
    for i,page in enumerate(pages,1):
        w=float(page.attrib['width']);h=float(page.attrib['height']);assert abs(min(w,h)-210/25.4*72)<1 and abs(max(w,h)-297/25.4*72)<1,(p,w,h)
        for word in page.iter('{http://www.w3.org/1999/xhtml}word'):
            if float(word.attrib['xMin'])<-.5 or float(word.attrib['yMin'])<-.5 or float(word.attrib['xMax'])>w+.5 or float(word.attrib['yMax'])>h+.5:off.append([str(p),i,word.text])
    pdf_checks[str(p.relative_to(base))]=len(pages)
 assert not off,off[:10]
 assert pdf_checks['main/Table_1_cohort.pdf']==pdf_checks['main/Table_2_core_comparisons.pdf']==1
 checks['a4_pdf_pages']=pdf_checks;checks['off_page_pdf_words']=0
 code=[base/name for name in ('rebuild_tables.py','rebuild_electronic_tables.py','workbook_export.py','rendering_support/table_assets.py')]
 for p in code:
    text=p.read_text();assert not any(v in text for v in ('/home/','/media/','/tmp/','from reporting','from results_recreation')),(p,'VM dependency')
 checks['absolute_vm_dependencies']=0;checks['workbook_validation']='OOXML package structure and all data cells independently round-tripped; desktop Office application not available on this VM.'
 checks['status']='passed'
 (base/'qa/validation.json').write_text(json.dumps(checks,indent=2)+'\n')
 return checks

if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
