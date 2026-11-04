"""Human reading copies of complete electronic supplements; raw source CSVs remain intact."""
from pathlib import Path
import importlib.util
import sys
import shutil
import pandas as pd
from rendering_support.table_assets import dataframe_to_markdown, dataframe_to_html, write_pdf_sections
P=Path(__file__).resolve().parent/'_build';S=P.parent/'inputs'
spec=importlib.util.spec_from_file_location('table_labels',Path(__file__).with_name('rebuild_tables.py'))
labels=importlib.util.module_from_spec(spec);spec.loader.exec_module(labels)
COHORTS={'main_shared':'Main shared cohort','icu_waveform_sensitivity':'Broader ICU cohort'}
OUTCOMES={'any_aki':'AKI within72hours','icu_admission':'ICU admission','mortality':'In-hospital mortality'}
OUTCOMES['any_aki']='AKI within 72 hours'
BRANCHES={'non_windowed':'Whole case','windowed':'Windowed'}
MODELS={'logistic_regression':'LR','Logistic regression':'LR','xgboost':'XGBoost','ebm':'EBM'}
METRICS={'auprc':'AP','auroc':'AUROC','brier':'Brier'}
import json
class FeatureLabels:
    def __init__(self):self.labels=json.loads((S/'feature_labels.json').read_text())
    def feature_label(self,key):return self.labels.get(key,key)
DICTIONARY=FeatureLabels()


def ordered_source(frame):
    """Fixed scientific identities first; thresholds/folds then local record order."""
    frame=frame.copy();columns=[]
    mappings={
      'cohort_scope':{'main_shared':0,'icu_waveform_sensitivity':1,'mortality_waveform_sensitivity':2},
      'outcome':{'any_aki':0,'icu_admission':1,'mortality':2},
      'branch':{'non_windowed':0,'windowed':1},
      'model_name':{'Logistic regression':0,'XGBoost':1,'EBM':2},
      'reference_model':{'logistic_regression':0,'xgboost':1,'ebm':2},
      'candidate_model':{'logistic_regression':0,'xgboost':1,'ebm':2},
      'feature_set':{key:i for i,key in enumerate(labels.INPUT_ORDER)},
      'reference_feature_set':{key:i for i,key in enumerate(labels.INPUT_ORDER)},
      'candidate_feature_set':{key:i for i,key in enumerate(labels.INPUT_ORDER)},
      'metric':{'auprc':0,'auroc':1,'brier':2}}
    for key,mapping in mappings.items():
        if key in frame:
            rank='_order_'+key;frame[rank]=frame[key].map(mapping).fillna(99);columns.append(rank)
    if 'fold' in frame:
        frame['_order_fold']=frame.fold.map(lambda value:int(str(value).removeprefix('outer_')));columns.append('_order_fold')
    for key in ('threshold_probability','native_rank'):
        if key in frame:columns.append(key)
    return frame.sort_values(columns,kind='stable').drop(columns=[c for c in frame if c.startswith('_order_')]) if columns else frame


def humanize(d):
    d=d.copy()
    maps={'cohort_scope':COHORTS,'outcome':OUTCOMES,'branch':BRANCHES,
          'reference_model':MODELS,'candidate_model':MODELS,'metric':METRICS,
          'feature_set':labels.NAMES,'reference_feature_set':labels.NAMES,'candidate_feature_set':labels.NAMES}
    for column,lookup in maps.items():
        if column in d:d[column]=d[column].map(lambda value:lookup.get(value,value))
    if 'fold' in d:
        d['fold']=d.fold.map(lambda value:'Fold '+str(int(str(value).removeprefix('outer_'))+1))
    for column in ('feature','label'):
        if column in d:d[column]=d[column].str.replace('Pleth —','PLETH —',regex=False)
    return d.rename(columns={'cohort_scope':'Cohort','outcome':'Outcome','branch':'Branch','metric':'Metric',
        'reference_model':'Reference model','candidate_model':'Candidate model',
        'feature_set':'Feature set','reference_feature_set':'Reference inputs','candidate_feature_set':'Candidate inputs',
        'fold':'Outer fold','feature':'Feature','native_rank':'Native rank','heldout_rank':'Held-out rank',
        'native_importance':'Native importance','heldout_mean_absolute':'Held-out mean absolute contribution',
        'missing_score':'Missing-bin score','missing_train_n':'Missing training n','train_n':'Training n',
        'missing_test_n':'Missing test n','test_n':'Test n','calibration_slope':'Calibration slope','label':'Feature'})


def export(stem,title,d,notes):
    d=d.fillna('')
    (P/(stem+'.md')).write_text('# '+title+'\n\n'+notes+'\n\n'+dataframe_to_markdown(d)+'\n')
    (P/(stem+'.html')).write_text(dataframe_to_html(d,title=title,notes=notes))
    d.to_csv(P/(stem+'_display.csv'),index=False)
    assert not d.astype(str).apply(lambda col:col.str.lower().eq('nan')).to_numpy().any()
    print(stem,len(d))


def interval(r,scale=1):
    return f'{scale*r.delta:.3f} ({scale*r.lower:.3f}–{scale*r.upper:.3f})'.replace('-','−')


def family_comparisons():
    """Complete S4 archive bundle, preserving the frozen numerical CSV verbatim."""
    d=ordered_source(pd.read_csv(S/'Table_S4_family_comparisons.csv'))
    t=d[['cohort_scope','outcome','branch','reference_feature_set','reference_model','candidate_model','metric']].copy()
    t['Reference estimate']=d.reference_estimate.map(lambda x:f'{x:.3f}')
    t['Candidate estimate']=d.candidate_estimate.map(lambda x:f'{x:.3f}')
    t['Difference (95% CI)']=d.apply(interval,axis=1)
    t=humanize(t).rename(columns={'Reference inputs':'Matched input set'})
    title='Table S4. Paired model comparisons'
    notes='Candidate-minus-reference differences (95% paired patient-cluster bootstrap CIs), matched by cohort, outcome, branch and input set. Negative Brier differences favor the candidate. AP, average precision; LR, logistic regression.'
    export('Table_S4_family_comparisons',title,t,notes)
    shutil.copyfile(S/'Table_S4_family_comparisons.csv',P/'Table_S4_family_comparisons_raw.csv')
    sections=[]
    for context,frame in t.groupby(['Cohort','Outcome','Branch'],sort=False):
        sections.append({'heading':' — '.join(context),'dataframe':frame.drop(columns=['Cohort','Outcome','Branch']),'notes':notes})
    write_pdf_sections(P/'Table_S4_family_comparisons.pdf',title=title,sections=sections)


def main():
    d=pd.read_csv(S/'Table_S1_descriptives_missingness.csv')
    t=d[['cohort_scope','branch','section','variable','category','summary','missing_n_percent_display']].copy()
    missing=d.row_type.eq('feature_missingness')
    t.loc[missing,'missing_n_percent_display']=d.loc[missing].apply(lambda r:f'{int(r.missing_n):,} ({r.missing_percent:.1f}%)',axis=1)
    t['variable']=t.variable.str.replace('10^3','10³',regex=False)
    # The category is nested under its parent variable, as in the older clinical table.
    t['category']=t.category.replace({'M':'Male','F':'Female'})
    t=humanize(t).rename(columns={'section':'Clinical section','variable':'Variable','category':'Category','summary':'Summary','missing_n_percent_display':'Missing n (%)'})
    export('Table_S1_descriptives_missingness','Table S1. Cohort characteristics and feature missingness',t,'Main shared cohort:3,571 operations/3,432 patients; broader ICU cohort:5,765/5,543. Continuous values are median(Q1,Q3); categorical values are n(%). Missingness is reported by cohort and branch.'.replace(':3',': 3').replace(':5',': 5').replace('median(Q1,Q3)','median (Q1, Q3)').replace('n(%)','n (%)'))
    family_comparisons()
    d=pd.read_csv(S/'absolute_core_source_rows.csv')
    d['_outcome']=d.outcome.map({'any_aki':0,'icu_admission':1})
    d['_model']=d.model_name.map({'Logistic regression':0,'XGBoost':1,'EBM':2})
    d['_input']=d.feature_set.map({'preop_only':0,'all_waveforms':1,'preop_and_all_waveforms':2})
    d=d.sort_values(['_outcome','_model','_input','threshold_probability'])
    t=pd.DataFrame({'Outcome':d.outcome.map(OUTCOMES),'Model':d.model_name.replace({'Logistic regression':'LR'}),'Feature set':d.feature_set.map(labels.NAMES),'Threshold (%)':(d.threshold_probability*100).map(lambda x:f'{x:.1f}')})
    for key,name in [('model','Model net benefit /100 (95% CI)'),('treat_all','Treat-all net benefit /100 (95% CI)')]:
        t[name]=d.apply(lambda r:f'{r[key]*100:.3f} ({r[key+"_lower"]*100:.3f}–{r[key+"_upper"]*100:.3f})'.replace('-','−'),axis=1)
    t['Treat-none net benefit /100']=(d.treat_none*100).map(lambda x:f'{x:.3f}')
    export('Table_S5A_absolute_net_benefit','Table S5A. Net benefit',t,'Whole-case main shared cohort. Net benefit per 100 operations (95% pointwise patient-cluster bootstrap CIs). Treat-none = 0.'.replace('by100','by 100').replace('pointwise95','pointwise 95').replace('is0','is 0'))
    d=ordered_source(pd.read_csv(S/'Table_S5_paired_decision_curve_values.csv'))
    t=d[['cohort_scope','outcome','branch','reference_model','candidate_model','reference_feature_set','candidate_feature_set']].copy()
    t['Threshold (%)']=(d.threshold_probability*100).map(lambda x:f'{x:.1f}')
    t['Δ net benefit /100 (95% CI)']=d.apply(lambda r:interval(r,100),axis=1)
    export('Table_S5_paired_decision_curve_values','Table S5B. Paired net-benefit differences',humanize(t),'Candidate-minus-reference net benefit per 100 operations (95% pointwise paired patient-cluster bootstrap CIs). LR, logistic regression.'.replace('by100','by 100').replace('pointwise95','pointwise 95'))
    d=ordered_source(pd.read_csv(S/'Table_S6_EBM_fold_importance.csv'))
    t=d.drop(columns=['family','group']).copy()
    t['feature']=t.feature.map(DICTIONARY.feature_label)
    for c in ('native_rank','heldout_rank'):t[c]=t[c].map(lambda x:str(int(x)))
    for c in ('native_importance','heldout_mean_absolute'):t[c]=t[c].map(lambda x:f'{x:.6f}')
    export('Table_S6_EBM_fold_importance','Table S6. EBM term importance',humanize(t),'Main cohort. Native importance uses training-distribution weights; held-out importance is the mean absolute uncalibrated term contribution in outer-test operations.'.replace('labels1','labels 1').replace('All11,294','All 11,294'))
    d=ordered_source(pd.read_csv(S/'Table_S6_selected_missing_bins.csv'))
    t=d[['outcome','branch','feature_set','fold','label','missing_score','missing_train_n','train_n','missing_test_n','test_n','calibration_slope']].copy()
    for c in ('missing_score','calibration_slope'):t[c]=t[c].map(lambda x:f'{x:.4f}')
    export('Table_S6B_selected_missing_bins','Table S6B. EBM missing-value scores and support',humanize(t),'Scores are uncalibrated log-odds contributions. Counts refer to the corresponding outer training or test fold.'.replace('labels1','labels 1').replace('folds0','folds 0'))

if __name__=='__main__':main()
