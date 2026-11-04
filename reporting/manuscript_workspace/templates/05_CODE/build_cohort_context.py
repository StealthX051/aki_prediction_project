"""Operation-level cohort context from the frozen roster and source laboratory data."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'02_TABLES'))
import rebuild_tables as tables
spec=importlib.util.spec_from_file_location('publication_tables',Path(__file__).with_name('build_publication_tables.py'))
publication=importlib.util.module_from_spec(spec);spec.loader.exec_module(publication)
OUT=ROOT/'01_MANUSCRIPT/publication_tables'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def n_percent(n,denominator):return f'{int(n):,} ({100*int(n)/int(denominator):.1f}%)'
def quantiles(series,digits=0):
    q=series.quantile([.25,.5,.75])
    return f'{q.loc[.5]:.{digits}f} ({q.loc[.25]:.{digits}f}, {q.loc[.75]:.{digits}f})',q.to_dict()


def main():
    master,context,source_hashes=tables.load_cohort_context()
    roster=master.caseid.to_list();n=len(roster)
    lab_path=ROOT/'04_DATA/raw_source/lab_data.csv'
    source_hashes[str(lab_path.relative_to(ROOT))]=digest(lab_path)
    labs=pd.read_csv(lab_path)
    labs=labs[labs.caseid.isin(roster)&labs.name.eq('cr')].copy()
    for key in ('dt','result'):labs[key]=pd.to_numeric(labs[key],errors='coerce')
    labs=labs[np.isfinite(labs.dt)&np.isfinite(labs.result)&labs.result.gt(0)]
    labs['opend']=labs.caseid.map(context.set_index('caseid').opend)
    labs['hours_postop']=(labs.dt-labs.opend)/3600
    labs=labs[labs.hours_postop.gt(0)&labs.hours_postop.le(72)]
    count72=labs.groupby('caseid').size().reindex(roster,fill_value=0)
    count48=labs[labs.hours_postop.le(48)].groupby('caseid').size().reindex(roster,fill_value=0)
    assert count72.gt(0).all() and int(count48.gt(0).sum())==3525
    context['postop_cr_measurements_72h']=context.caseid.map(count72)
    context['postop_cr_measurements_48h']=context.caseid.map(count48)
    case_columns=['caseid','subjectid','optype','aki_label','aki_stage','y_icu_admit','preop_cr','preop_egfr_ckdepi_2021','operative_duration_minutes','postop_cr_measurements_48h','postop_cr_measurements_72h']
    context[case_columns].to_csv(OUT/'Table_S7_cohort_context_case_data.csv',index=False)
    grouped=context.groupby('optype',sort=True).agg(n_operations=('caseid','size'),aki_n=('aki_label','sum'),icu_n=('y_icu_admit','sum')).reset_index()
    assert len(grouped)==11 and int(grouped.n_operations.sum())==3571 and int(grouped.aki_n.sum())==189 and int(grouped.icu_n.sum())==834
    grouped[['aki_n','icu_n']]=grouped[['aki_n','icu_n']].astype(int)
    procedure=pd.DataFrame({'Procedure':grouped.optype,
        'Operations n (%)':grouped.n_operations.map(lambda x:n_percent(x,n)),
        'AKI events n (%)':grouped.apply(lambda r:n_percent(r.aki_n,r.n_operations),axis=1),
        'ICU admissions n (%)':grouped.apply(lambda r:n_percent(r.icu_n,r.n_operations),axis=1)})
    procedure.loc[len(procedure)]=['All operations',n_percent(n,n),n_percent(context.aki_label.sum(),n),n_percent(context.y_icu_admit.sum(),n)]
    raw=[]
    for r in grouped.itertuples(index=False):raw.append({'section':'Surgical case mix','characteristic':r.optype,'n_operations':int(r.n_operations),'aki_n':int(r.aki_n),'icu_n':int(r.icu_n),'denominator_n':int(r.n_operations),'denominator_type':'operations within procedure','source_fields':'optype; aki_label; y_icu_admit'})
    duration,q_duration=quantiles(context.operative_duration_minutes)
    cr,q_cr=quantiles(master.preop_cr,2)
    egfr,q_egfr=quantiles(master.preop_egfr_ckdepi_2021,1)
    measurements,q_measurements=quantiles(count72)
    repeated_n=int(master.subjectid.value_counts().gt(1).sum());n_patients=int(master.subjectid.nunique())
    assert repeated_n==120 and n_patients==3432
    records=[
        ('Patients with multiple operations',n_percent(repeated_n,n_patients),'',repeated_n,n_patients,'patients','subjectid'),
        ('Operative duration (min)',duration,'0 (0.0%)',None,n,'operations','(opend − opstart) / 60'),
        ('Preoperative creatinine (mg/dL)',cr,'0 (0.0%)',None,n,'operations','preop_cr'),
        ('Preoperative eGFR (mL/min/1.73 m²)',egfr,n_percent(master.preop_egfr_ckdepi_2021.isna().sum(),n),None,n,'operations','preop_egfr_ckdepi_2021'),
        ('eGFR <60 mL/min/1.73 m²',n_percent(master.preop_egfr_ckdepi_2021.lt(60).sum(),n),n_percent(master.preop_egfr_ckdepi_2021.isna().sum(),n),int(master.preop_egfr_ckdepi_2021.lt(60).sum()),n,'operations','preop_egfr_ckdepi_2021 < 60'),
        ('No AKI',n_percent(context.aki_stage.eq(0).sum(),n),'',int(context.aki_stage.eq(0).sum()),n,'operations','aki_stage == 0'),
        ('AKI stage 1',n_percent(context.aki_stage.eq(1).sum(),n),'',int(context.aki_stage.eq(1).sum()),n,'operations','aki_stage == 1'),
        ('AKI stage 2',n_percent(context.aki_stage.eq(2).sum(),n),'',int(context.aki_stage.eq(2).sum()),n,'operations','aki_stage == 2'),
        ('AKI stage 3',n_percent(context.aki_stage.eq(3).sum(),n),'',int(context.aki_stage.eq(3).sum()),n,'operations','aki_stage == 3'),
        ('Valid creatinine within 48 h after operation end',n_percent(count48.gt(0).sum(),n),'',int(count48.gt(0).sum()),n,'operations','finite positive cr; opend < dt ≤ opend + 48 h'),
        ('Valid creatinine within 72 h after operation end',n_percent(count72.gt(0).sum(),n),'',int(count72.gt(0).sum()),n,'operations','finite positive cr; opend < dt ≤ opend + 72 h'),
        ('Creatinine measurements per operation within 72 h',measurements,'',None,n,'operations','count finite positive cr; opend < dt ≤ opend + 72 h')]
    context_display=pd.DataFrame([[r[0],r[1],r[2]] for r in records],columns=['Characteristic','Finding','Missing n (%)'])
    context_display['Characteristic']=context_display.Characteristic.replace({'Patients with multiple operations':'Patients with repeat operations', 'Valid creatinine within 48 h after operation end':'Postoperative creatinine by 48 h', 'Valid creatinine within 72 h after operation end':'Postoperative creatinine by 72 h', 'Creatinine measurements per operation within 72 h':'Creatinine results per operation (72 h)'})
    for label,value,missing,count,denominator,kind,fields in records:
        record={'section':'Clinical and ascertainment context','characteristic':label,'value':value,'missing_n_percent':missing,'count':count,'denominator_n':denominator,'denominator_type':kind,'source_fields':fields}
        q={'Operative duration (min)':q_duration,'Preoperative creatinine (mg/dL)':q_cr,'Preoperative eGFR (mL/min/1.73 m²)':q_egfr,'Creatinine measurements per operation within 72 h':q_measurements}.get(label)
        if q:record.update(q1=float(q[.25]),median=float(q[.5]),q3=float(q[.75]))
        raw.append(record)
    raw=pd.DataFrame(raw)
    display=pd.concat([procedure.assign(Section='Surgical case mix'),context_display.assign(Section='Clinical and ascertainment context')],ignore_index=True)
    section1={'heading':'Surgical case mix and outcomes','pdf_title':'Table S7. Surgical case mix and outcomes','dataframe':procedure,
        'styles':{procedure.index[-1]:{column:{'bold':True,'top_rule':True} for column in procedure}},
        'pdf_notes':'Operation percentages use all 3,571 operations; outcome percentages use operations within each procedure. AKI, acute kidney injury; ICU, intensive care unit.'}
    section2={'heading':'Renal function, AKI severity and ascertainment','pdf_title':'Table S7 (continued). Renal function, AKI severity and ascertainment','dataframe':context_display,
        'pdf_notes':'Values are median (Q1, Q3) or n (%). Repeat-patient percentages use 3,432 patients; other percentages use 3,571 operations. Creatinine windows start at operation end; results must be finite and positive. eGFR, estimated glomerular filtration rate.'}
    notes='Main cohort: 3,571 operations, 3,432 patients. Values are median (Q1, Q3) or n (%). Outcome percentages are within procedure; repeat-patient percentages use patients. AKI, acute kidney injury; ICU, intensive care unit; eGFR, estimated glomerular filtration rate.'
    publication.bundle('Table_S7_cohort_context','Table S7. Cohort context',raw,display,[section1,section2],notes,
        'All 11 procedure categories from the exact frozen main-shared whole-case roster, with operation-level outcome counts; source-defined AKI stages and finite-positive postoperative creatinine measurements are summarized without relabeling outcomes.',list(source_hashes))
    assert all(digest(ROOT/name)==value for name,value in source_hashes.items())
    record=publication.INDEX[-1];record['source_sha256']=source_hashes
    (OUT/'cohort_context_table_index.json').write_text(json.dumps({'tables':[record],'models_fitted':0,'outcome_labels_changed':False,'source_hashes_unchanged':True},indent=2)+'\n')
    audit={'source_sha256':source_hashes,'source_files_unchanged':True,'operations':n,'patients':n_patients,'patients_with_multiple_operations':repeated_n,'procedure_counts_sum':int(grouped.n_operations.sum()),'procedure_aki_counts_sum':int(grouped.aki_n.sum()),'procedure_icu_counts_sum':int(grouped.icu_n.sum()),'aki_stage_counts':{str(stage):int(context.aki_stage.eq(stage).sum()) for stage in range(4)},'creatinine_48h_coverage_operations':int(count48.gt(0).sum()),'creatinine_72h_coverage_operations':int(count72.gt(0).sum()),'postop_creatinine_72h_quantiles':{str(k):float(v) for k,v in q_measurements.items()},'pdf_pages':record['pdf_pages'],'raw_rows':len(raw),'display_rows_by_section':[len(procedure),len(context_display)]}
    (ROOT/'02_TABLES/qa/cohort_context_supplement_review.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
