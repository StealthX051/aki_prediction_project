"""One-command rebuild of the current manuscript displays from bundled data."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
STEPS=[
 '03_FIGURES/paper/rebuild_paper_figures.py',
 '03_FIGURES/single_channel_comparison/rebuild_s1.py',
 '03_FIGURES/decision_curve/rebuild_dca.py',
 '05_CODE/rebuild_narrative_figures.py',
 '05_CODE/rebuild_roc_supplement.py',
 '05_CODE/rebuild_case_figure.py',
 '02_TABLES/rebuild_tables.py',
 '05_CODE/build_cohort_context.py',
 '05_CODE/build_calibration_support.py',
 '05_CODE/build_waveform_descriptor_key.py',
 '05_CODE/build_publication_tables.py',
 '05_CODE/build_manuscript_documents.py',
 '05_CODE/build_story_map.py',
 '05_CODE/build_publication_proofs.py',
 '02_TABLES/validate_tables.py',
 '05_CODE/validate_publication_style.py',
 '05_CODE/catalogue_reader_text.py',
 '05_CODE/validate_publication_navigation.py']

def refresh_inventory():
 from publication_style import STYLE_SHA256, STYLE, WORD_FONT, FONT_FAMILY
 table_root=ROOT/'02_TABLES'
 index_path=table_root/'table_index.json'
 index=json.loads(index_path.read_text());index['style_profile_sha256']=STYLE_SHA256
 index['office_font_family']=WORD_FONT;index_path.write_text(json.dumps(index,indent=2)+'\n')
 files=[p for p in sorted(table_root.rglob('*')) if p.is_file() and p.name!='file_manifest.json' and '__pycache__' not in p.parts]
 (table_root/'file_manifest.json').write_text(json.dumps({'style_profile_sha256':STYLE_SHA256,'files':[{'file':str(p.relative_to(table_root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]},indent=2)+'\n')
 record={'schema_version':STYLE['schema_version'],'profile':'05_CODE/publication_style.json','profile_sha256':STYLE_SHA256,'helper_sha256':hashlib.sha256((ROOT/'05_CODE/publication_style.py').read_bytes()).hexdigest(),'graphics_font':FONT_FAMILY,'office_font':WORD_FONT,'current_generated_figures':21,'publication_figures':15,'source_repository_profile':'reporting/publication_style.json','historical_source_snapshots_restyled':False}
 (ROOT/'07_PROVENANCE/shared_publication_style.json').write_text(json.dumps(record,indent=2)+'\n')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--package',action='store_true',help='Also create the reading and complete-workspace ZIPs.')
 args=parser.parse_args()
 profile=ROOT/'05_CODE/publication_style.json';before=hashlib.sha256(profile.read_bytes()).hexdigest()
 for i,script in enumerate(STEPS,1):
  print(f'[{i}/{len(STEPS)}] {script}',flush=True)
  subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)
 assert hashlib.sha256(profile.read_bytes()).hexdigest()==before,'Style profile changed during rebuild; rerun with one fixed profile.'
 (ROOT/'00_START_HERE/qa/shared_style_rebuild.json').write_text(json.dumps({'status':'passed','profile_sha256':before,'steps':STEPS,'model_fits':0,'bootstrap_draws':0},indent=2)+'\n')
 refresh_inventory()
 if args.package:
  for script in ['05_CODE/package_publication.py','05_CODE/package_workspace.py']:
   subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)
 print('All current manuscript displays rebuilt from one appearance profile.',flush=True)
if __name__=='__main__':main()
