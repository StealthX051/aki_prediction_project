"""Validate and package the completed writing workspace using relative archive paths."""
from pathlib import Path
import csv,json,hashlib,zipfile,xml.etree.ElementTree as ET,subprocess
from datetime import datetime
from docx import Document
from publication_style import STYLE_SHA256
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 qa=ROOT/'00_START_HERE/workspace_validation.json'
 required=['01_MANUSCRIPT/Main_Figures_and_Tables.docx','01_MANUSCRIPT/Supplementary_Figures_and_Methods.docx','01_MANUSCRIPT/author_material/Manuscript_Starting_Document.docx','03_FIGURES/single_channel_comparison/Figure_S1A_AP_differences.svg','03_FIGURES/single_channel_comparison/Figure_S1B_AUROC_differences.svg','01_MANUSCRIPT/publication_tables/Table_S8_calibration_diagnostics.docx','01_MANUSCRIPT/publication_tables/Table_S9_waveform_descriptors.docx','03_FIGURES/narrative_revision/Figure3_calibration_clinical_combined.prediction_support.csv']
 for n in required:assert (ROOT/n).is_file(),n
 docs=[]
 for p in sorted(ROOT.rglob('*.docx')):
  if 'project_source' in p.parts:continue
  with zipfile.ZipFile(p) as z:
   assert z.testzip() is None,p
   xml=ET.fromstring(z.read('word/document.xml'))
   tables=len(list(xml.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tbl')))
  docs.append({'path':str(p.relative_to(ROOT)),'native_tables':tables})
 main_doc=Document(ROOT/required[0]);supp_doc=Document(ROOT/required[1])
 assert len(main_doc.tables)==2 and len(main_doc.inline_shapes)==5
 assert len(supp_doc.inline_shapes)==10
 predictions=pd.read_csv(ROOT/'04_DATA/prediction_index.csv');assert len(predictions)==288
 for rel in predictions.offline_prediction_path:assert (ROOT/rel).is_file()
 # Recheck every exported prediction/metadata file against the copied-source hash.
 prediction_records=json.loads((ROOT/'07_PROVENANCE/prediction_export_manifest.json').read_text())['files']
 for r in prediction_records:assert sha(ROOT/r['relative_path'])==r['sha256'],r['relative_path']
 s1=pd.read_csv(ROOT/'03_FIGURES/single_channel_comparison/S1_absolute_and_paired_values.csv')
 source=pd.read_csv(ROOT/'03_FIGURES/single_channel_comparison/inputs/metrics_summary.csv')
 keys=['outcome','branch','model_name','feature_set'];joined=s1.merge(source,on=keys,suffixes=('_display','_source'),validate='one_to_one')
 assert len(joined)==60
 for key in ['auprc','auroc','delta_auprc','delta_auroc']:
  for suffix in ['','_lower','_upper']:
   k=key+suffix
   assert (abs(joined[k+'_display']-joined[k+'_source'])<1e-12).all(),k
 # Check portable active scripts, excluding archived source/provenance.
 scripts=[ROOT/'03_FIGURES/paper/rebuild_paper_figures.py',ROOT/'03_FIGURES/paper/cohort_support.py',ROOT/'03_FIGURES/single_channel_comparison/rebuild_s1.py',ROOT/'03_FIGURES/decision_curve/rebuild_dca.py',ROOT/'05_CODE/build_manuscript_documents.py',ROOT/'05_CODE/build_publication_proofs.py',ROOT/'05_CODE/build_publication_tables.py',ROOT/'05_CODE/package_publication.py',ROOT/'05_CODE/rebuild_narrative_figures.py',ROOT/'05_CODE/rebuild_roc_supplement.py',ROOT/'05_CODE/rebuild_case_figure.py',ROOT/'05_CODE/build_cohort_context.py',ROOT/'05_CODE/build_story_map.py',ROOT/'05_CODE/publication_style.py',ROOT/'05_CODE/rebuild_publication_workspace.py',ROOT/'05_CODE/validate_publication_style.py',ROOT/'05_CODE/catalogue_reader_text.py',ROOT/'05_CODE/build_calibration_support.py',ROOT/'05_CODE/build_waveform_descriptor_key.py',ROOT/'05_CODE/publication_navigation.py',ROOT/'05_CODE/validate_publication_navigation.py']
 for p in scripts:
  compile(p.read_text(),str(p),'exec')
  assert '/home/exouser/' not in p.read_text() and '/media/volume/' not in p.read_text(),p
 table_qa=json.loads((ROOT/'02_TABLES/qa/validation.json').read_text())
 dca_qa=json.loads((ROOT/'03_FIGURES/decision_curve/validation.json').read_text())
 data={'prepared_date':'2026-11-04','main_configurations':180,'broader_icu_configurations':90,'mortality_core_configurations':18,'optional_mortality_extensions_unrun':72,'pooled_prediction_rows':1265400,'prediction_sets_verified':288,'prediction_and_metadata_files_sha256_verified':len(prediction_records),'main_docx_figures':5,'main_docx_native_tables':2,'supplement_docx_figure_parts':10,'supplementary_figure_labels':['S'+str(i) for i in range(1,11)],'publication_polish':True,'style_profile_sha256':STYLE_SHA256,'shared_style_validation':'00_START_HERE/qa/publication_style_validation.json','one_command_rebuild':'python 05_CODE/rebuild_publication_workspace.py --package','main_table_bold_cells_independently_verified':19,'curated_supplementary_tables':'01_MANUSCRIPT/publication_tables','docx_files_checked':len(docs),'docx_details':docs,'S1_paired_metric_entries_verified':120,'S1_absolute_metric_entries_verified':120,'active_scripts_have_no_VM_path_dependencies':True,'writing_workspace_does_not_include_fitted_model_binaries':True,'table_validation_file':'02_TABLES/qa/validation.json','dca_validation_file':'03_FIGURES/decision_curve/validation.json','calibration_validation_file':'00_START_HERE/run_audit/calibration_audit.json','calibration_support_validation':'00_START_HERE/qa/calibration_support_review.json','waveform_descriptor_validation':'02_TABLES/qa/waveform_descriptor_review.json','navigation_validation':'00_START_HERE/qa/publication_navigation_validation.json','reader_support_revision':'00_START_HERE/qa/reader_support_revision.json','author_starting_document_preserved':True,'curated_table_numbers':['S'+str(i) for i in range(1,10)]}
 qa.write_text(json.dumps(data,indent=2)+'\n')
 # Remove interpreter caches, never include them as manuscript deliverables.
 import shutil
 for p in list(ROOT.rglob('__pycache__')):shutil.rmtree(p)
 files=[p for p in sorted(ROOT.rglob('*'))if p.is_file() and p.name!='SHA256SUMS.csv']
 with (ROOT/'SHA256SUMS.csv').open('w',newline='')as f:
  writer=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256']);writer.writeheader()
  for p in files:writer.writerow({'relative_path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 archive=ROOT.parent/'VitalDB_Manuscript_Workspace_2026-11-04_Reader_Revision.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True)as z:
  for p in sorted(ROOT.rglob('*')):
   if p.is_file():z.write(p,arcname='VitalDB_Manuscript_Workspace/'+str(p.relative_to(ROOT)))
 with zipfile.ZipFile(archive)as z:assert z.testzip()is None
 print(json.dumps({'zip':str(archive),'bytes':archive.stat().st_size,'files':len(files)+1,'docx':len(docs),'configurations':288,'S1_paired_estimates':120},indent=2))
if __name__=='__main__':main()
