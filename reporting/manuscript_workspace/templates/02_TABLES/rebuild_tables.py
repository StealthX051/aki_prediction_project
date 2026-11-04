"""Rebuild publication tables from immutable saved summaries; no fitting or bootstrap."""
from pathlib import Path
import hashlib, json, re, shutil, sys
from html import escape
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.section import WD_ORIENT
import textwrap
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_CODE'))
from publication_style import apply_document_style, WORD_TABLE_PT, WORD_FONT, color, style_context
from rendering_support.table_assets import export_dataframe_bundle, write_docx_sections, write_markdown_sections, write_pdf_sections, dataframe_to_html, _add_dataframe_to_doc, numeric_display_columns
METRIC_DISPLAY_MAP={'auprc':'AUPRC','auroc':'AUROC','brier':'Brier Score','sensitivity':'Sensitivity','specificity':'Specificity','f1':'F1 Score','precision':'Precision','accuracy':'Accuracy'}
def prepare_metrics_for_display(raw):
    frame=raw.rename(columns={'outcome':'Outcome','branch':'Branch','model_name':'Model'}).copy()
    frame['Feature Set Key']=frame['feature_set']
    return frame
# Emphasis is descriptive and follows the numbers readers actually see.
def get_style_info(frame):
    result={index:{} for index in frame.index}
    for key,label in METRIC_DISPLAY_MAP.items():
        if key not in frame:continue
        shown=frame[key].map(lambda value:float(f"{value:.3f}") if pd.notna(value) else float('nan'))
        best=shown.min() if key=='brier' else shown.max()
        for index in frame.index:
            if pd.notna(shown.loc[index]) and shown.loc[index]==best:
                result[index][label]={'bold':True}
    return result
def get_delta_style_info(frame, mapping):return {}
HERE = Path(__file__).resolve().parent
OUT = HERE / '_build'
DATA = HERE / 'inputs'
OUT.mkdir(exist_ok=True)
SOURCES = {}
METRICS = ('auprc','auroc','brier','sensitivity','specificity','precision','f1','accuracy')
LABELS = {**METRIC_DISPLAY_MAP, 'auprc':'AP', 'brier':'Brier', 'f1':'F1'}
OUTCOMES = {'any_aki':'AKI within 72 hours','icu_admission':'ICU admission','mortality':'In-hospital mortality'}
BRANCHES = {'non_windowed':'Whole case','windowed':'Windowed'}
FAMILIES = ['Logistic regression','XGBoost','EBM']
NAMES = {'preop_only':'Clinical only','all_waveforms':'Waveforms only','preop_and_all_waveforms':'Clinical + waveforms', 'awp_only':'AWP only','co2_only':'CO₂ only','ecg_only':'ECG only','pleth_only':'PLETH only'}
for channel in ('awp','co2','ecg','pleth'):
    label={'awp':'AWP','co2':'CO₂','ecg':'ECG','pleth':'PLETH'}[channel]
    NAMES[f'preop_and_{channel}']=f'Clinical + {label}'
    NAMES[f'preop_and_all_minus_{channel}']=f'Clinical + waveforms except {label}'
BOLD_KEY = 'Bold identifies the highest displayed estimate within each outcome, branch and model family (lowest for Brier).'
NOTE = ('Estimates (95% patient-cluster bootstrap CIs). Classification uses training-selected, fold-specific thresholds. '
        +BOLD_KEY+' AP, average precision; AUROC, area under the receiver operating characteristic curve.')
DELTA_NOTE = ('Paired input-set minus clinical-only differences (95% patient-cluster bootstrap CIs), matched by outcome, branch and model. '
              'Differences use the 0–1 scale; negative Brier differences favor the candidate.')



INPUT_ORDER=['preop_only','all_waveforms','preop_and_all_waveforms']+[
    f'{channel}_only' for channel in ('ecg','pleth','co2','awp')]+[
    f'preop_and_{channel}' for channel in ('ecg','pleth','co2','awp')]+[
    f'preop_and_all_minus_{channel}' for channel in ('ecg','pleth','co2','awp')]

def publication_input_order(key):return INPUT_ORDER.index(key)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def load(study,scope):
    path=DATA/f'source_{scope}_metrics_summary.csv'
    frame=pd.read_csv(path)
    assert not frame.duplicated(['outcome','branch','model_name','feature_set']).any()
    return frame

def fmt(row,key):
    return f'{row[key]:.3f} ({row[key+"_lower"]:.3f}–{row[key+"_upper"]:.3f})'.replace('-','−')

def merge_html(path,title,parts,notes=None):
    chunks=[]
    for p in parts:
        txt=p.read_text()
        match=re.search(r'<h1>(.*?)</h1>(.*)',txt,flags=re.S)
        assert match,p
        chunks.append('<section id="section-'+str(len(chunks))+'"><h2>'+match.group(1)+'</h2>'+match.group(2)+'</section>')
    css='body{font:10pt Georgia,serif;color:#222;margin:24px}h1{font-size:16pt}h2{font-size:12pt;margin-top:2em}p{font-size:9pt;max-width:110ch;line-height:1.5}table{border-collapse:collapse;width:100%;font-size:9pt}th,td{padding:7px 9px;border-bottom:1px solid #ddd}thead th{border-top:1px solid #333;border-bottom:1px solid #777}td{font-variant-numeric:tabular-nums}@media print{h1,h2{break-after:avoid}thead{display:table-header-group}tr{break-inside:avoid}}'
    css=css.replace('Georgia,serif',WORD_FONT+',sans-serif').replace('#222',color('ink')).replace('#ddd',color('table_rule')).replace('#333',color('ink')).replace('#777',color('reference'))
    navigation=''
    if len(chunks)>=6:
        from publication_navigation import short_label
        links=[]
        for i,chunk in enumerate(chunks):
            heading=re.search(r'<h2>(.*?)</h2>',chunk,re.S).group(1)
            links.append('<a href="#section-'+str(i)+'">'+short_label(heading)+'</a>')
        navigation='<nav aria-label="Table sections"><p>'+(' · '.join(links))+'</p></nav>'
    path.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>'+escape(title)+'</title><style>'+css+'</style></head><body><h1>'+escape(title)+'</h1><p>'+escape(notes or '')+'</p>'+navigation+''.join(chunks)+'</body></html>',encoding='utf-8')


def fragment(base,title,display,notes,styles):
    base.parent.mkdir(exist_ok=True)
    path=base.with_suffix('.html')
    path.write_text(dataframe_to_html(display,title=title,notes=notes,styles=styles))
    return path


def fixed_pdf(path,title,frame,notes,*,portrait=False,styles=None,separators=()):
    """Two main tables get deliberate A4 composition instead of generic pagination."""
    width,height=(8.27,11.69) if portrait else (11.69,8.27)
    fractions=[.50,.31,.19] if portrait else [.085,.09,.225,.15,.15,.15,.15]
    assert len(fractions)==len(frame.columns)
    font=9.2 if portrait else 9
    data=[]
    for row in frame.itertuples(index=False,name=None):
        cells=[]
        for column,value,fraction in zip(frame.columns,row,fractions):
            text=str(value).replace(' (','\n(') if not portrait else str(value)
            if not portrait and column=='Outcome':text=text.replace('ICU admission','ICU\nadmission')
            n=max(12,int((width-.6)*fraction*72/(font*.52)))
            cells.append('\n'.join(textwrap.fill(line,n,break_long_words=False) for line in text.splitlines()))
        data.append(cells)
    headers=[str(c).replace('Δ AP vs clinical','Δ AP vs\nclinical') for c in frame.columns]
    row_heights=[max(c.count('\n')+1 for c in r)*font*1.18/72+.055 for r in [headers]+data]
    total=sum(row_heights);top=height-.62;bottom=top-total
    note_lines=textwrap.wrap(notes,110 if portrait else 165)
    assert bottom-(len(note_lines)*8.2*1.2/72+.3)>.15,(title,bottom,len(note_lines))
    with style_context():
        fig=plt.figure(figsize=(width,height));fig.patch.set_facecolor('white')
        fig.text(.3/width,1-.20/height,title,fontsize=12,weight='bold',va='top')
        ax=fig.add_axes([.3/width,bottom/height,(width-.6)/width,total/height]);ax.axis('off')
        table=ax.table(cellText=data,colLabels=headers,colWidths=fractions,cellLoc='right',bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(font)
        numeric=numeric_display_columns(frame)
        for (r,c),cell in table.get_celld().items():
            cell.set_height(row_heights[r]/total);cell.set_linewidth(0)
            cell.set_facecolor(color('table_header') if r==0 else (styles or {}).get(frame.index[r-1],{}).get(frame.columns[c],{}).get('bg','white'))
            cell.set_text_props(ha='right' if frame.columns[c] in numeric else 'left',weight='bold' if r==0 or (styles or {}).get(frame.index[r-1],{}).get(frame.columns[c],{}).get('bold') else 'normal')
            cell.PAD=.035
        y=1
        for r,h in enumerate(row_heights):
            ax.plot([0,1],[y,y],transform=ax.transAxes,color=color('reference') if r in {0,1} or r-1 in separators else color('table_rule'),lw=.65 if r in {0,1} or r-1 in separators else .25,clip_on=False)
            y-=h/total
        ax.plot([0,1],[0,0],transform=ax.transAxes,color=color('reference'),lw=.65,clip_on=False)
        fig.text(.3/width,(bottom-.18)/height,'\n'.join(note_lines),fontsize=8.2,va='top',linespacing=1.25)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        for cell in table.get_celld().values():
            text=cell.get_text().get_window_extent(renderer);box=cell.get_window_extent(renderer)
            assert box.x0-.2<=text.x0 and text.x1<=box.x1+.2 and box.y0-.2<=text.y0 and text.y1<=box.y1+.2,(title,cell.get_text().get_text())
        fig.savefig(path)
        plt.close(fig)
    from publication_navigation import add_pdf_navigation
    record=add_pdf_navigation(path,title,[(title,1)])
    path.with_suffix('.navigation.json').write_text(json.dumps(record,indent=2)+'\n')


def main_docx(path,title,frame,notes,styles=None,portrait=False):
    doc=apply_document_style(Document(), body_size=WORD_TABLE_PT);section=doc.sections[0]
    if portrait:section.page_width=Inches(8.27);section.page_height=Inches(11.69)
    else:section.orientation=WD_ORIENT.LANDSCAPE;section.page_width=Inches(11.69);section.page_height=Inches(8.27)
    section.top_margin=section.bottom_margin=Inches(.45)
    section.left_margin=section.right_margin=Inches(.5)
    doc.add_heading(title,1)
    display=frame.map(lambda value:str(value).replace(' (','\n(')) if not portrait else frame
    _add_dataframe_to_doc(doc,display,styles)
    doc.add_paragraph(notes);doc.save(path)


def complete_bundle(raw,stem,title,delta=False,exploratory=False):
    pd.DataFrame(raw).to_csv(OUT/(stem+'_raw.csv'),index=False)
    prepared=prepare_metrics_for_display(raw)
    sections=[]; fragments=[]; displays=[]
    for outcome in [key for key in OUTCOMES if key in set(raw.outcome)]:
      for branch in ('non_windowed','windowed'):
       for family in FAMILIES:
        frame=prepared[(prepared.Outcome==outcome)&(prepared.Branch==branch)&(prepared.Model==family)].copy()
        if frame.empty: continue
        frame['_order']=frame['Feature Set Key'].map(publication_input_order)
        frame=frame.sort_values('_order')
        display=pd.DataFrame({'Feature set':frame['Feature Set Key'].map(NAMES)},index=frame.index)
        assert display['Feature set'].notna().all()
        original_styles=get_delta_style_info(frame,{f'Δ {label}':f'delta_{key}' for key,label in METRIC_DISPLAY_MAP.items()}) if delta else get_style_info(frame)
        styles={i:{} for i in frame.index}
        for key in METRICS:
            source_key=('delta_' if delta else '')+key
            label=('Δ ' if delta else '')+LABELS[key]
            display[label]=frame.apply(lambda row: 'Reference' if delta and row['Feature Set Key']=='preop_only' else fmt(row,source_key),axis=1)
            source_label=('Δ ' if delta else '')+METRIC_DISPLAY_MAP[key]
            for i in frame.index:
                if source_label in original_styles.get(i,{}): styles[i][label]=original_styles[i][source_label]
        pop=f'{int(frame.n.iloc[0]):,} operations; {int(frame.n_patients.iloc[0]):,} patients; event prevalence {frame.prevalence.iloc[0]:.2%}.'
        assert frame.n.nunique()==frame.n_patients.nunique()==frame.prevalence.nunique()==1
        heading=f'{OUTCOMES[outcome]} — {BRANCHES[branch]} — {family}'
        notes=pop
        if exploratory: notes='52 positive operations from 47 patients. '+notes
        section={'heading':heading,'dataframe':display,'styles':styles,'notes':notes}
        sections.append(section)
        target=OUT/'sections'/f'{stem}_{outcome}_{branch}_{family.lower().replace(" ","_")}'
        fragments.append(fragment(target,heading,display,notes,styles))
        full=display.copy()
        full.insert(0,'Model',family);full.insert(0,'Branch',BRANCHES[branch]);full.insert(0,'Outcome',OUTCOMES[outcome])
        displays.append(full)
    pd.concat(displays,ignore_index=True).to_csv(OUT/(stem+'_display.csv'),index=False)
    shared_notes=DELTA_NOTE if delta else NOTE
    if exploratory:shared_notes='Exploratory mortality analysis. '+shared_notes
    book_sections=[{'heading':'Table key','text':shared_notes}]+sections
    write_markdown_sections(OUT/(stem+'.md'),title=title,sections=book_sections)
    write_docx_sections(OUT/(stem+'.docx'),title=title,sections=book_sections)
    pdf_sections=[]
    short_note='Estimates (95% patient-cluster bootstrap CIs). Bold identifies the highest displayed value within each outcome/model (lowest for Brier).'
    if delta:short_note='Paired input-set minus clinical-only differences (95% patient-cluster bootstrap CIs). Negative Brier differences indicate improvement.'
    grouped=[]
    if len(raw)<=36:
        for outcome in [key for key in OUTCOMES if key in set(raw.outcome)]:
          for branch in ('non_windowed','windowed'):
            matching=[v for v in sections if v['heading'].startswith(OUTCOMES[outcome]+' — '+BRANCHES[branch]+' — ')]
            if not matching:continue
            frames=[];combined_styles={}
            for v in matching:
                f=v['dataframe'].copy();f.insert(0,'Model',v['heading'].split(' — ')[-1].replace('Logistic regression','LR'));frames.append(f);combined_styles.update(v['styles'])
            grouped.append({'heading':OUTCOMES[outcome]+' — '+BRANCHES[branch],'dataframe':pd.concat(frames),'styles':combined_styles,'notes':matching[0]['notes']})
    else:grouped=sections
    for v in grouped:
        identities=[c for c in v['dataframe'] if c in {'Model','Feature set'}]
        for name,keys in [('Discrimination and Brier score',('auprc','auroc','brier')),('Classification at fold thresholds',('sensitivity','specificity','precision','f1','accuracy'))]:
            columns=identities+[('Δ ' if delta else '')+LABELS[k] for k in keys]
            pdf_sections.append({**v,'heading':v['heading']+' — '+name,'dataframe':v['dataframe'][columns],'notes':v['notes']+' '+short_note})
    write_pdf_sections(OUT/(stem+'.pdf'),title=title,sections=pdf_sections)
    merge_html(OUT/(stem+'.html'),title,fragments,shared_notes)
    return {'rows':len(raw),'metrics':list(METRICS),'sections':len(sections),'paired_differences':delta,'exploratory':exploratory}


def compact_main(raw):
    core=raw[raw.branch.eq('non_windowed') & raw.feature_set.isin(['preop_only','all_waveforms','preop_and_all_waveforms'])]
    parts=[];styles={}
    for outcome in ('any_aki','icu_admission'):
      frame=prepare_metrics_for_display(core[core.outcome.eq(outcome)])
      frame['_order']=frame['Feature Set Key'].map(publication_input_order)
      frame['_family']=frame.Model.map({v:i for i,v in enumerate(FAMILIES)})
      frame=frame.sort_values(['_family','_order'])
      display=pd.DataFrame({'Outcome':'AKI' if outcome=='any_aki' else 'ICU admission','Model':frame.Model.replace({'Logistic regression':'LR'}),'Feature set':frame['Feature Set Key'].map(NAMES)},index=frame.index)
      for _,g in frame.groupby('Model'):
        for i,cells in get_style_info(g).items():
          styles[i]={LABELS[k]:cells[METRIC_DISPLAY_MAP[k]] for k in ('auprc','auroc','brier') if METRIC_DISPLAY_MAP[k] in cells}
      for key in ('auprc','auroc','brier','delta_auprc'):
        display['Δ AP vs clinical' if key=='delta_auprc' else LABELS[key]]=frame.apply(lambda r:'Reference' if key=='delta_auprc' and r['Feature Set Key']=='preop_only' else fmt(r,key),axis=1)
      parts.append(display)
    display=pd.concat(parts)
    notes=('Estimates (95% patient-cluster bootstrap CIs); 3,571 operations, 3,432 patients. '
           'AKI, 5.3%; ICU admission, 23.4%. Δ AP = input set minus same-model clinical-only AP (0–1 scale). '
           'Bold identifies the highest displayed AP/AUROC and lowest Brier within each outcome/model. '
           'AKI, acute kidney injury; ICU, intensive care unit; AP, average precision; AUROC, area under the receiver operating characteristic curve; LR, logistic regression; EBM, explainable boosting machine.')
    title='Table 2. Whole-case model performance'
    view=display.copy()
    view.loc[view.Outcome.eq(view.Outcome.shift()),'Outcome']=''
    repeated_family=display.Model.eq(display.Model.shift()) & display.Outcome.eq(display.Outcome.shift())
    view.loc[repeated_family,'Model']=''
    for position,index in enumerate(view.index):
        if position%3==0:
            for column in view:
                styles.setdefault(index,{}).setdefault(column,{})['top_rule']=True
            for column in ('Outcome','Model'):styles[index][column]['bold']=True
    export_dataframe_bundle(OUT/'Table_2_core_comparisons',title=title,dataframe=display.reset_index(drop=True),display_dataframe=view,notes=notes,styles=styles)
    fixed_pdf(OUT/'Table_2_core_comparisons.pdf',title,view,notes,styles=styles,separators=range(0,18,3))
    main_docx(OUT/'Table_2_core_comparisons.docx',title,view,notes,styles)


def load_cohort_context():
    """Exact frozen main roster; operative times are seconds from recording start."""
    root=HERE.parent
    files={name:root/name for name in (
        '04_DATA/processed/cohort_rosters.json',
        '04_DATA/processed/aki_features_master_wide.csv',
        '04_DATA/processed/aki_pleth_ecg_co2_awp.csv',
        '04_DATA/raw_source/clinical_parameters.csv')}
    hashes={name:sha(path) for name,path in files.items()}
    roster=json.loads(files['04_DATA/processed/cohort_rosters.json'].read_text())['main_shared']['branches']['non_windowed']['caseids']
    assert len(roster)==len(set(roster))==3571
    master=pd.read_csv(files['04_DATA/processed/aki_features_master_wide.csv']).set_index('caseid').loc[roster].reset_index()
    context=pd.read_csv(files['04_DATA/processed/aki_pleth_ecg_co2_awp.csv']).set_index('caseid').loc[roster].reset_index()
    assert master.caseid.equals(context.caseid) and master.subjectid.equals(context.subjectid)
    assert master.subjectid.nunique()==3432 and master.aki_label.sum()==189 and master.y_icu_admit.sum()==834
    assert master.aki_label.equals(context.aki_label) and master.y_icu_admit.equals(context.y_icu_admit)
    dictionary=pd.read_csv(files['04_DATA/raw_source/clinical_parameters.csv']).set_index('Parameter')
    assert dictionary.loc['opstart','Unit']==dictionary.loc['opend','Unit']=='sec'
    context['operative_duration_minutes']=(context.opend-context.opstart)/60
    assert context.operative_duration_minutes.notna().all() and context.operative_duration_minutes.gt(0).all()
    context['preop_egfr_ckdepi_2021']=master.preop_egfr_ckdepi_2021
    assert context.aki_stage.isin([0,1,2,3]).all()
    return master,context,hashes


def main_cohort_table(cohort):
    master,context,hashes=load_cohort_context()
    duration=context.operative_duration_minutes.quantile([.25,.5,.75])
    egfr=master.preop_egfr_ckdepi_2021
    low_egfr=int(egfr.lt(60).sum());stage23=int(context.aki_stage.ge(2).sum())
    n=len(master)
    additions={
        'AKI within 72 hours':{'Characteristic':'Stage 2–3 AKI','Finding':f'{stage23:,} ({100*stage23/n:.1f}%)','Missing n (%)':''},
        'Open approach':{'Characteristic':'Operative duration (min)','Finding':f'{duration.loc[.5]:.0f} ({duration.loc[.25]:.0f}, {duration.loc[.75]:.0f})','Missing n (%)':'0 (0.0%)'},
        'eGFR (mL/min/1.73 m²)':{'Characteristic':'eGFR <60 mL/min/1.73 m²','Finding':f'{low_egfr:,} ({100*low_egfr/n:.1f}%)','Missing n (%)':f'{egfr.isna().sum():,} ({100*egfr.isna().mean():.1f}%)'}}
    rows=[];styles={};raw_rows=[]
    groups=[('Study population',range(0,4)),('Demographics',range(4,7)),('Operative characteristics',range(7,13)),('Comorbidities',range(13,15)),('Procedure type',range(15,26)),('Preoperative laboratory values',range(26,33))]
    for heading,indices in groups:
        index=len(rows);rows.append([heading,'','0 (0.0%)' if heading=='Procedure type' else ''])
        styles[index]={c:{'bg':color('table_band'),'bold':True} for c in ('Characteristic','Finding (N = 3,571)','Missing n (%)')}
        for i in indices:
            r=cohort.iloc[i];label=r.Characteristic;missing=r['Missing n (%)'];original_label=label
            if label.startswith('Procedure type:'):label='\u2003'+label.split(': ',1)[1];missing=''
            if label.startswith('ASA physical status '):label='ASA '+{'1':'I','2':'II','3':'III','4':'IV'}[label[-1]]
            label=label.replace('Hb (','Hemoglobin (').replace('10^3','10³')
            rows.append([label,r.Finding,missing]);raw_rows.append(r.to_dict())
            if original_label in additions:
                extra=additions[original_label]
                prefix='\u2003' if original_label=='AKI within 72 hours' else ''
                rows.append([prefix+extra['Characteristic'],extra['Finding'],extra['Missing n (%)']]);raw_rows.append(extra)
    display=pd.DataFrame(rows,columns=['Characteristic','Finding (N = 3,571)','Missing n (%)']).fillna('')
    precisions={'Age (years)':0,'BMI (kg/m²)':1,'Hemoglobin (g/dL)':1,'Creatinine (mg/dL)':2,'eGFR (mL/min/1.73 m²)':1,'BUN (mg/dL)':0,'Albumin (g/dL)':1,'WBC (10³/µL)':2,'CRP (mg/dL)':2}
    for i,row in display.iterrows():
        label=row['Characteristic']
        if label in precisions:
            values=re.findall(r'[-+]?\d+(?:\.\d+)?',row['Finding (N = 3,571)'])
            assert len(values)==3,(label,values)
            values=[f'{float(value):.{precisions[label]}f}' for value in values]
            display.at[i,'Finding (N = 3,571)']=values[0]+' ('+values[1]+', '+values[2]+')'
    notes=('Values are median (Q1, Q3) or n (%); percentages use operations. '
           'Procedure missingness appears at its heading. '
           'AKI, acute kidney injury; ICU, intensive care unit; ASA, American Society of Anesthesiologists physical status; BMI, body mass index; BUN, blood urea nitrogen; eGFR, estimated glomerular filtration rate; WBC, white blood cell count; CRP, C-reactive protein.')
    title='Table 1. Cohort characteristics'
    export_dataframe_bundle(OUT/'Table_1_cohort',title=title,dataframe=pd.DataFrame(raw_rows),display_dataframe=display,notes=notes,styles=styles)
    display.to_csv(OUT/'Table_1_cohort_display.csv',index=False)
    fixed_pdf(OUT/'Table_1_cohort.pdf',title,display,notes,portrait=True,styles=styles)
    main_docx(OUT/'Table_1_cohort.docx',title,display,notes,styles,portrait=True)
    audit={'source_sha256':hashes,'operations':n,'patients':int(master.subjectid.nunique()),'selected_caseids':master.caseid.tolist(),
           'additions':list(additions.values()),'duration_quantiles_minutes':{str(k):float(v) for k,v in duration.items()},
           'egfr_below60_n':low_egfr,'egfr_missing_n':int(egfr.isna().sum()),'stage2_3_aki_n':stage23,
           'original_rows_preserved':len(cohort),'raw_rows':len(raw_rows),'display_rows':len(display),
           'all_percentages_use_operations_except_unique_patient_count':True,
           'operative_duration_is_descriptive_context_not_an_added_model_predictor':True,'no_model_fitting_or_outcome_relabeling':True}
    assert all(sha(HERE.parent/name)==value for name,value in hashes.items())
    (HERE/'qa/cohort_narrative_review.json').write_text(json.dumps(audit,indent=2)+'\n')


def input_key():
    DEFAULT_FEATURE_SET_MAP={}
    channels=['ecg','pleth','co2','awp']
    for key in INPUT_ORDER:
        components=(['preop'] if key.startswith('preop') else [])
        if key in {'all_waveforms','preop_and_all_waveforms'}:components+=channels
        elif 'all_minus_' in key:components+=[c for c in channels if c!=key.split('all_minus_')[1]]
        elif key!='preop_only':components+=[next(c for c in channels if key.endswith('_'+c) or key==c+'_only')]
        DEFAULT_FEATURE_SET_MAP[key]=components
    rows=[]
    for key in sorted(NAMES,key=publication_input_order):
        components=DEFAULT_FEATURE_SET_MAP[key]
        rows.append([NAMES[key]]+['✓' if component in components else '—' for component in ('preop','ecg','pleth','co2','awp')])
    display=pd.DataFrame(rows,columns=['Feature set','Clinical','ECG','PLETH','CO₂','AWP'])
    export_dataframe_bundle(OUT/'Table_S2_input_key',title='Input-set key (Tables S2–S3)',dataframe=display,notes='✓ Included; — omitted. Clinical denotes preoperative predictors. ECG, electrocardiogram; PLETH, photoplethysmogram; CO₂, carbon dioxide; AWP, airway pressure.')


def main():
    main=load('manuscript-supplementary-v1','main_shared')
    broader=load('manuscript-icu-supplementary-v1','icu_waveform_sensitivity')
    mortality=load('manuscript-optimized-v1','mortality_waveform_sensitivity')
    assert (len(main),len(broader),len(mortality))==(180,90,18)
    inventory={}
    core=main[main.feature_set.isin(['preop_only','all_waveforms','preop_and_all_waveforms'])]
    inventory['S2_core']=complete_bundle(core,'Table_S2_core_all_metrics','Core model performance')
    inventory['S2_full']=complete_bundle(main,'Table_S2_main_all_configurations','Table S2. Model performance in the main cohort')
    inventory['S2_delta']=complete_bundle(main,'Table_S2B_main_paired_differences','Table S2B. Paired differences from clinical-only models',delta=True)
    inventory['S3_full']=complete_bundle(broader,'Table_S3_broader_ICU_all_configurations','Table S3. Model performance in the broader ICU cohort')
    inventory['S3_delta']=complete_bundle(broader,'Table_S3B_broader_ICU_paired_differences','Table S3B. Broader ICU cohort: paired differences',delta=True)
    inventory['mortality_backup']=complete_bundle(mortality,'Meeting_backup_mortality_all_metrics','Exploratory mortality model performance',exploratory=True)
    compact_main(main)
    main_cohort_table(pd.read_csv(DATA/'Table_1_cohort.csv'))
    input_key()
    shutil.rmtree(OUT/'sections')
    from rebuild_electronic_tables import main as electronic_main
    electronic_main()
    organize_outputs()
    from workbook_export import build_workbooks
    build_workbooks(HERE)
    (HERE/'qa/build_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(inventory,indent=2))

def organize_outputs():
    for path in OUT.iterdir():
        folder='main' if path.name.startswith(('Table_1_','Table_2_')) else 'meeting_backup' if path.name.startswith('Meeting_') else 'supplement'
        path.replace(HERE/folder/path.name)
    OUT.rmdir()

if __name__=='__main__':main()
