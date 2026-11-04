"""Export exact saved-fold EBM case contributions; never fit or select on labels.

Model-dependent export is optional and needs the original verified fold bundles.
Use rebuild_case_figure.py for portable figure regeneration from the exported CSVs.
"""
from pathlib import Path
import argparse,gc,hashlib,json,pickle,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'03_FIGURES/heldout_cases'
OUTCOMES={'any_aki':'AKI','icu_admission':'ICU admission'}
CHANNELS={'SNUADC_ECG_II_':'ECG','SNUADC_PLETH_':'PLETH','Primus_CO2_':'CO₂','Primus_AWP_':'AWP'}
CLINICAL={'sex':'Sex','emop':'Emergency status','department':'Surgical department','approach':'Surgical approach','asa':'ASA physical status','optype':'Procedure type','ane_type':'Anesthesia type','preop_htn':'Hypertension','preop_dm':'Diabetes','preop_ecg':'Preoperative ECG','preop_pft':'Pulmonary function','inpatient_preop':'Preoperative inpatient status'}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--project-root',type=Path,required=True,help='Repository providing model_creation for trusted saved objects')
 parser.add_argument('--model-results-root',type=Path,help='Optional relocated nested_cv root containing models/; otherwise use original locations from prediction_index.csv')
 args=parser.parse_args();sys.path.insert(0,str(args.project_root.resolve()))
 from model_creation.validation import is_outer_fold_checkpoint_complete
 OUT.mkdir(parents=True,exist_ok=True)
 matrix_path=ROOT/'04_DATA/processed/aki_features_master_wide.csv';dataset_sha=sha(matrix_path)
 matrix=pd.read_csv(matrix_path);assert not matrix.caseid.duplicated().any()
 lookup=matrix.set_index('caseid')
 index=pd.read_csv(ROOT/'04_DATA/prediction_index.csv')
 labels=json.loads((ROOT/'02_TABLES/inputs/feature_labels.json').read_text())
 ledger=[];terms=[];groups=[];contexts=[];audit=[];source_records={str(matrix_path.relative_to(ROOT)):dataset_sha}
 case_number=0
 for outcome,outcome_label in OUTCOMES.items():
  mask=index.cohort_scope.eq('main_shared')&index.model.eq('ebm')&index.outcome.eq(outcome)&index.branch.eq('non_windowed')
  combined_row=index[mask&index.feature_set.eq('preop_and_all_waveforms')].iloc[0]
  clinical_row=index[mask&index.feature_set.eq('preop_only')].iloc[0]
  combined_path=ROOT/combined_row.offline_prediction_path;clinical_path=ROOT/clinical_row.offline_prediction_path
  combined=pd.read_csv(combined_path);clinical=pd.read_csv(clinical_path).set_index('caseid')
  assert len(combined)==3571 and combined.caseid.is_unique and set(combined.caseid)==set(clinical.index)
  source_records[str(combined_path.relative_to(ROOT))]=sha(combined_path)
  source_records[str(clinical_path.relative_to(ROOT))]=sha(clinical_path)
  configuration=(args.model_results_root/'models/ebm'/outcome/'non_windowed/preop_and_all_waveforms') if args.model_results_root else Path(combined_row.source_configuration_path)
  for quantile in (0.1,0.9):
   # Selection sees only operation ID, held-out probability and fold. Outcomes are joined afterward.
   target=float(np.quantile(combined.y_prob_calibrated,quantile))
   blind=combined[['caseid','y_prob_calibrated','outer_fold_id']].copy()
   blind['distance']=(blind.y_prob_calibrated-target).abs()
   chosen=blind.sort_values(['distance','caseid'],kind='stable').iloc[0]
   caseid=int(chosen.caseid);fold=int(chosen.outer_fold_id);display_case='Case '+chr(65+case_number);case_number+=1
   saved=combined[combined.caseid.eq(caseid)].iloc[0]
   fold_dir=configuration/'artifacts/folds/repeat_00'/f'outer_{fold:02d}'
   metadata=json.loads((fold_dir/'metadata.json').read_text())
   assert metadata['status']=='completed' and metadata['validation_fingerprint']['dataset_hash']==dataset_sha
   assert is_outer_fold_checkpoint_complete(fold_dir),str(fold_dir)
   assert sha(fold_dir/'model.pkl')==metadata['model_sha256']
   assert sha(fold_dir/'preprocessor.pkl')==metadata['preprocessor_sha256']
   heldout=pd.read_csv(fold_dir/'test_predictions.csv').set_index('caseid')
   assert caseid in heldout.index and int(heldout.loc[caseid].outer_fold_id)==fold
   training=pd.read_csv(fold_dir/'train_oof.csv',usecols=['caseid','subjectid'])
   assert caseid not in set(training.caseid)
   assert int(saved.subjectid) not in set(training.subjectid)
   for key in ('y_prob_raw','y_prob_calibrated'):
    assert abs(float(heldout.loc[caseid,key])-float(saved[key]))<1e-14
   with (fold_dir/'preprocessor.pkl').open('rb') as stream:preprocessor=pickle.load(stream)
   with (fold_dir/'model.pkl').open('rb') as stream:model=pickle.load(stream)
   raw=lookup.loc[[caseid],list(preprocessor.numeric_columns_)+list(preprocessor.categorical_columns_)]
   transformed=preprocessor.transform(raw)
   assert int(clinical.loc[caseid].outer_fold_id)==fold
   for clinical_feature in raw.columns:
    if any(clinical_feature.startswith(prefix) for prefix in CHANNELS):continue
    value=raw.iloc[0][clinical_feature]
    contexts.append({'display_case':display_case,'outcome':outcome,'feature':clinical_feature,'raw_value':None if pd.isna(value) else value,'label':CLINICAL.get(clinical_feature,labels.get(clinical_feature,clinical_feature))})
   assert all(len(v)==1 for v in model.term_features_),'Additive EBM required'
   contribution=np.asarray(model.eval_terms(transformed))[0]
   baseline=float(np.asarray(model.intercept_).ravel()[0])
   raw_logit=float(baseline+contribution.sum());raw_probability=float(model.predict_proba(transformed)[0,1])
   assert abs(raw_probability-float(saved.y_prob_raw))<1e-12
   assert abs(raw_logit-float(np.asarray(model.decision_function(transformed)).ravel()[0]))<1e-12
   calibration=json.loads((fold_dir/'calibration.json').read_text());a=float(calibration['intercept']);c=float(calibration['slope']);eps=float(calibration['eps'])
   assert eps<raw_probability<1-eps,'A clipping residual must be handled explicitly before plotting this case'
   calibrated_baseline=a+c*baseline;calibrated=1/(1+np.exp(-(calibrated_baseline+c*contribution.sum())))
   assert abs(calibrated-float(saved.y_prob_calibrated))<1e-12
   grouped={};members={};types={}
   for j,(feature,value) in enumerate(zip(model.term_names_,contribution)):
    feature=str(feature);feature_column=str(transformed.columns[model.term_features_[j][0]])
    channel=next((value for prefix,value in CHANNELS.items() if feature_column.startswith(prefix)),None)
    if channel:group=channel;kind='waveform'
    else:
     group=next((column for column in sorted(preprocessor.categorical_columns_,key=len,reverse=True) if feature_column.startswith(column+'_')),feature_column)
     kind='clinical'
    grouped[group]=grouped.get(group,0.0)+float(c*value);members.setdefault(group,[]).append(feature);types[group]=kind
    terms.append({'display_case':display_case,'outcome':outcome,'caseid':caseid,'fold':fold,'feature':feature,'transformed_value':float(transformed.iloc[0,model.term_features_[j][0]]),'original_group':group,'kind':kind,'raw_term_contribution':float(value),'calibrated_logit_contribution':float(c*value),'label':labels.get(feature,feature)})
   clinical_keys=sorted([key for key in grouped if types[key]=='clinical'],key=lambda key:(-abs(grouped[key]),key))
   top=clinical_keys[:3];remaining=clinical_keys[3:]
   readable=[]
   for key in top:
    readable.append((key,CLINICAL.get(key,labels.get(key,key)),grouped[key],members[key],'clinical'))
   readable.append(('other_clinical','Other clinical terms',sum(grouped[k] for k in remaining),sum((members[k] for k in remaining),[]),'clinical'))
   for channel in CHANNELS.values():
    readable.append((channel,channel,grouped.get(channel,0.0),members.get(channel,[]),'waveform'))
   position=calibrated_baseline
   for order,(key,label,value,original_members,kind) in enumerate(readable,1):
    groups.append({'display_case':display_case,'outcome':outcome,'order':order,'group':key,'label':label,'kind':kind,'contribution':float(value),'start':float(position),'end':float(position+value),'member_count':len(original_members),'members':json.dumps(original_members)})
    position+=value
   assert abs(position-(calibrated_baseline+c*contribution.sum()))<1e-12
   # Outcome is added only after the blind selection and exact reconstruction.
   actual=int(saved.y_true);clinical_probability=float(clinical.loc[caseid].y_prob_calibrated)
   ledger.append({'display_case':display_case,'outcome':outcome,'outcome_label':outcome_label,'target_quantile':quantile,'target_probability':target,'absolute_distance':float(chosen.distance),'caseid':caseid,'outer_fold_id':fold,'observed_outcome':actual,'clinical_probability':clinical_probability,'combined_probability':float(saved.y_prob_calibrated),'raw_probability':raw_probability,'raw_intercept':baseline,'calibration_intercept':a,'calibration_slope':c,'calibration_eps':eps,'calibrated_baseline':calibrated_baseline,'calibrated_final_logit':float(position),'selection_consulted_outcome':False})
   audit.append({'display_case':display_case,'source_fold':str(fold_dir),'dataset_sha256':dataset_sha,'model_sha256':metadata['model_sha256'],'preprocessor_sha256':metadata['preprocessor_sha256'],'calibration_sha256':sha(fold_dir/'calibration.json'),'complete_checkpoint_verified':True,'heldout_case_verified':True,'heldout_patient_excluded_from_training':True,'raw_probability_absolute_error':abs(raw_probability-float(saved.y_prob_raw)),'calibrated_probability_absolute_error':abs(calibrated-float(saved.y_prob_calibrated)),'group_reconstruction_absolute_error':abs(position-(calibrated_baseline+c*contribution.sum())),'probability_clipping_active':False,'terms':len(contribution)})
   del model,preprocessor,transformed;gc.collect()
 pd.DataFrame(ledger).to_csv(OUT/'case_selection_ledger.csv',index=False)
 pd.DataFrame(terms).to_csv(OUT/'selected_native_and_calibrated_terms.csv',index=False)
 pd.DataFrame(groups).to_csv(OUT/'grouped_calibrated_contributions.csv',index=False)
 pd.DataFrame(contexts).to_csv(OUT/'case_clinical_context.csv',index=False)
 pd.DataFrame(audit).to_csv(OUT/'case_reconstruction_audit.csv',index=False)
 (OUT/'export_audit.json').write_text(json.dumps({'status':'passed','operations':4,'models_fitted':0,'bootstrap_draws_generated':0,'case_selection':'Nearest pooled held-out combined-EBM risk to the 10th and 90th empirical percentiles within each outcome; ascending operation ID breaks ties; outcomes and contribution patterns excluded from selection.','probability_clipping_active':False,'max_calibrated_probability_error':max(row['calibrated_probability_absolute_error'] for row in audit),'source_sha256':source_records,'case_checks':audit},indent=2)+'\n')
 (OUT/'README.txt').write_text('Held-out EBM case illustrations\n\nFour operations selected without examining outcomes or contribution patterns: nearest the 10th and 90th percentiles of pooled held-out combined-model risk within each outcome; operation ID resolves ties. Selection IDs remain in the ledger; the artwork uses Case A–D.\n\nEach case uses its saved outer-fold model, preprocessing and recalibrator. Native signed term contributions are transformed to calibrated log-odds by the saved calibration slope. Baseline is calibration intercept plus slope times model intercept. The top three clinical feature groups, remaining clinical terms and four waveform-channel sums reconstruct the saved prediction exactly. One-hot terms are summed within their original clinical feature. No fitting or bootstrap resampling occurs.\n\nClinical-only risk is a separately fitted EBM prediction for the same held-out operation; it is shown as a comparator, not an additive component.\n\nPortable figure rebuilding: python 05_CODE/rebuild_case_figure.py. Re-exporting native terms additionally needs the original saved models and research environment; export_heldout_cases.py --help describes the optional model-dependent step. The CSVs and figure rebuild need no fitted models.\n')
 print(json.dumps({'cases':len(ledger),'terms':len(terms),'maximum_probability_error':max(r['calibrated_probability_absolute_error'] for r in audit)},indent=2))
if __name__=='__main__':main()
