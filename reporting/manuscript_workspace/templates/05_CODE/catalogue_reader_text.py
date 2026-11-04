"""Collect rendered reader text for editorial review; never alter scientific sources."""
from pathlib import Path
import importlib.util,json,re,xml.etree.ElementTree as ET
from docx import Document
ROOT=Path(__file__).resolve().parents[1]
MANIFESTS=['03_FIGURES/paper/figure_manifest.json','03_FIGURES/single_channel_comparison/figure_manifest.json','03_FIGURES/decision_curve/figures/dca_manifest.json','03_FIGURES/narrative_revision/figure_manifest.json','03_FIGURES/roc_curves/figure_manifest.json','03_FIGURES/heldout_cases/figure_manifest.json']
def plain(text):return re.sub(r'\s+',' ',text).strip()
def main():
 figures=[]
 for rel in MANIFESTS:
  path=ROOT/rel
  for item in json.loads(path.read_text())['figures']:
   base=path.parent/item['stem'];xml=ET.fromstring(base.with_suffix('.svg').read_text())
   text=[plain(' '.join(element.itertext())) for element in xml.iter('{http://www.w3.org/2000/svg}text')]
   figures.append({'stem':item['stem'],'title':item['title'],'caption':(path.parent/item['caption_file']).read_text().strip(),'rendered_text':list(dict.fromkeys(t for t in text if t))})
 tables=[]
 paths=sorted((ROOT/'02_TABLES').glob('*/*.docx'))+sorted((ROOT/'01_MANUSCRIPT/publication_tables').glob('*.docx'))
 for path in paths:
  document=Document(path)
  outside=[plain(p.text) for p in document.paragraphs if p.text.strip()]
  headers=list(dict.fromkeys(tuple(plain(c.text) for c in table.rows[0].cells) for table in document.tables))
  tables.append({'path':str(path.relative_to(ROOT)),'titles_headings_notes':outside,'column_headers':[list(h) for h in headers]})
 spec=importlib.util.spec_from_file_location('documents',Path(__file__).with_name('build_manuscript_documents.py'))
 D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
 main_figures,supp,_=D.entries()
 active=[{'label':f['label'],'title':f['title'],'caption':D.caption_text(f)} for f in main_figures+supp]
 inputs=[{'path':str(p.relative_to(ROOT)),'caption':p.read_text().strip()} for p in sorted((ROOT/'03_FIGURES/paper/inputs').rglob('*.txt')) if 'caption' in p.name]
 result={'figures':figures,'tables':tables,'active_figure_captions':active,'source_caption_inputs':inputs,'scope':'Current and reference generated reader text; historical source documents and machine identifiers retain their original text.'}
 (ROOT/'00_START_HERE/qa/reader_text_inventory.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print('Reader text catalogued:',len(figures),'figure sets;',len(tables),'native table documents')
if __name__=='__main__':main()
