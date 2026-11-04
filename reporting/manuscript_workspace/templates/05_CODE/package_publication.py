"""Package a clean publication reading/editing set, separate from the full source archive."""
from pathlib import Path
import csv,hashlib,importlib.util,json,shutil,zipfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent/'publication-packet'
DOCS=['Main_Figures_and_Tables.docx','Main_Figures_and_Tables.pdf','Supplementary_Figures_and_Methods.docx','Supplementary_Figures_and_Methods.pdf','Supplementary_Methods.docx','Supplementary_Methods.pdf','Supplementary_Methods.txt','Figure_Captions.docx','Supplementary_Tables_Index.docx','Figure_Table_Story_Map.docx','Figure_Table_Story_Map.txt','story_map.json']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(source,dest):dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
def relocate_table_index_links(path):
 """Keep relative table links valid in the smaller packet's folder layout."""
 part='word/_rels/document.xml.rels'
 with zipfile.ZipFile(path) as archive:
  files=[(item,archive.read(item.filename)) for item in archive.infolist()]
 root=ET.fromstring(next(data for item,data in files if item.filename==part))
 count=0
 for relation in root:
  target=relation.attrib.get('Target','')
  if target.startswith('publication_tables/'):
   relation.set('Target','../03_Supplementary_Tables/'+target.removeprefix('publication_tables/'));count+=1
 if count:
  temporary=path.with_suffix('.tmp.docx')
  with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
   for item,data in files:
    archive.writestr(item,ET.tostring(root,encoding='utf-8',xml_declaration=True) if item.filename==part else data)
  temporary.replace(path)
 return count

def main():
 # Replace only this script's dedicated generated staging directory.
 if OUT.exists():shutil.rmtree(OUT)
 OUT.mkdir()
 for name in DOCS:copy(ROOT/'01_MANUSCRIPT'/name,OUT/'01_Documents'/name)
 relocated_links=relocate_table_index_links(OUT/'01_Documents/Supplementary_Tables_Index.docx')
 for part,folder in [('main','02_Main_Tables'),('supplement','03_Supplementary_Tables')]:
  source=ROOT/'02_TABLES/main' if part=='main' else ROOT/'01_MANUSCRIPT/publication_tables'
  for p in sorted(source.iterdir()):
   if p.is_file() and p.suffix in {'.docx','.pdf','.html','.md','.csv','.json','.txt'}:copy(p,OUT/folder/p.name)
 spec=importlib.util.spec_from_file_location('documents',Path(__file__).with_name('build_manuscript_documents.py'))
 D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
 main_fig,supp,_=D.entries();index=[]
 for f in main_fig+supp:
  alias=f['label'].replace('Figure ','Fig');folder=ROOT/f['folder']
  for ext in ['svg','pdf','eps','png','tiff','jpg']:copy(folder/(f['stem']+'.'+ext),OUT/'04_Figures'/alias/(alias+'.'+ext))
  caption=f['label'].replace('Figure','Fig.')+' '+D.caption_text(f).rstrip('.')+'\n'
  (OUT/'04_Figures'/alias/(alias+'_caption.txt')).write_text(caption)
  copy(ROOT/'01_MANUSCRIPT/figure_proofs'/(f['stem']+'.pdf'),OUT/'04_Figures'/alias/(alias+'_captioned_proof.pdf'))
  data=folder/(f['stem']+'.plotted.csv')
  if not data.exists() and 'plotted_data_file' in f:data=folder/f['plotted_data_file']
  if not data.exists() and 'single_channel' in f['folder']:data=folder/'S1_absolute_and_paired_values.csv'
  assert data.exists(),f
  copy(data,OUT/'04_Figures'/alias/(alias+'_plotted_data.csv'))
  if f.get('support_data_file'):
   support_name='prediction_support' if f.get('support_kind','').startswith('held-out') else 'training_support'
   copy(folder/f['support_data_file'],OUT/'04_Figures'/alias/(alias+'_'+support_name+'.csv'))
  if f['stem']=='heldout_ebm_cases':
   for support in sorted(folder.glob('*.csv')):copy(support,OUT/'04_Figures'/alias/support.name)
  if f['stem']=='figure_4_ebm_response_functions':
   for key in ['figure_4_training_support.csv','figure_4_missing_bins.csv']:copy(folder/key,OUT/'04_Figures'/alias/key.replace('figure_4',alias))
  index.append({'figure':f['label'],'title':f['title'],'publication_alias':alias,'original_source_stem':f['stem'],'source_folder_in_complete_workspace':f['folder'],'data_sha256':sha(data)})
 (OUT/'Figure_Index.json').write_text(json.dumps(index,indent=2)+'\n')
 (OUT/'READ_ME_FIRST.txt').write_text("""VitalDB — publication packet, 4 November 2026

Start with 01_Documents/Figure_Table_Story_Map.docx for the argument and display
sequence. Main_Figures_and_Tables contains five figures and two native tables in
reading order: Fig1, Table1, Table2, Fig2, Fig3, Fig4, Fig5. Supplementary Word/PDF
books contain methods and ten sequential figures (S1–S10). Separate native Word,
PDF, HTML, Markdown and CSV supplementary tables accompany them.

Main displays: cohort; absolute core performance; paired AP/AUROC differences;
held-out calibration; native EBM global importance; selected response functions.
Table1 adds operative duration, preoperative eGFR below60, and stage2–3AKI.
Table S7 gives surgical case mix and ascertainment; Table S8 gives calibration
diagnostics; Table S9 defines the waveform descriptors displayed in the EBM figures.
Long-table PDFs have clickable section links and bookmarks. Word books and tables
provide navigation links.

Supporting figures: S1 PR; S2 ROC; S3/S4 channel AP/AUROC differences;
S5 windowed; S6 broader ICU; S7 conventional DCA; S8 paired net benefit;
S9 extended observed-value EBM functions; S10 four held-out local explanations.
Standalone missing-bin dotplots are excluded; support scores/counts remain in
TableS6 and the complete archive. Full eight-metric performance tables are retained.

04_Figures has numbered aliases Fig1–Fig5 and FigS1–FigS10, each with vector
SVG/PDF/EPS, PNG/TIFF/JPG, an editable caption, data and a captioned A4 proof.
Training-support CSVs accompany response functions; predicted-risk counts
accompany the calibration figure. S10 includes exact signed
native/calibrated terms, case selection and clinical context. Its cases illustrate
model reasoning rather than improvement; local channel contributions are not the
difference between separately fitted clinical and combined model risks.

PDF proofs are native vector compositions, not Microsoft Word layout renders.
Word captions and table cells are editable; vector artwork supports figure edits.
Table2 best displayed values remain bold, including displayed ties.

The complete manuscript-workspace ZIP additionally contains all imported
data, source documents, 288 prediction sets, exhaustive results, author draft,
audits and portable rebuild code. Fitted model binaries and original waveforms
remain outside the writing export. Included local-case figures rebuild from CSV;
selecting new cases requires the research models. No models were refitted and no
new bootstrap estimates were generated. No cloud upload was performed.
SHA256SUMS.csv checks all files. Earlier ZIP versions remain preserved.
""")
 copy(ROOT/'05_CODE/PUBLICATION_STYLE.md',OUT/'Style_Guide.md')
 copy(ROOT/'05_CODE/publication_style.json',OUT/'publication_style.json')
 files=[p for p in sorted(OUT.rglob('*')) if p.is_file()]
 with (OUT/'SHA256SUMS.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=['relative_path','bytes','sha256']);writer.writeheader()
  for p in files:writer.writerow({'relative_path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 archive=ROOT.parent/'VitalDB_Publication_Packet_2026-11-04_Reader_Revision.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(OUT.rglob('*')):
   if p.is_file():z.write(p,'VitalDB_Publication_Packet/'+str(p.relative_to(OUT)))
 with zipfile.ZipFile(archive) as z:assert z.testzip()is None
 record={'archive':archive.name,'bytes':archive.stat().st_size,'files':len(files)+1,'main_figures':len(main_fig),'supplementary_figures':len(supp),'source_mapping':'Figure_Index.json','relocated_table_index_links':relocated_links,'style_profile_sha256':D.STYLE_SHA256}
 (ROOT/'00_START_HERE/qa/publication_packet.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record,indent=2))
if __name__=='__main__':main()
