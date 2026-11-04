"""Check actual publication outputs against the shared appearance profile."""
from pathlib import Path
import importlib.util,json,re,xml.etree.ElementTree as ET
from docx import Document
from PIL import Image
import pandas as pd
from publication_style import (STYLE_SHA256,STYLE,FONT_FAMILY,WIDTH_MM,PNG_DPI,TIFF_DPI,JPG_DPI,
                               WORD_FONT,WORD_TABLE_PT,WORD_BODY_PT,color)
ROOT=Path(__file__).resolve().parents[1]
MANIFESTS=['03_FIGURES/paper/figure_manifest.json','03_FIGURES/single_channel_comparison/figure_manifest.json',
 '03_FIGURES/decision_curve/figures/dca_manifest.json','03_FIGURES/narrative_revision/figure_manifest.json',
 '03_FIGURES/roc_curves/figure_manifest.json','03_FIGURES/heldout_cases/figure_manifest.json']

def style_check(document,table=False):
 for name in ['Normal','Title','Heading 1','Heading 2','Heading 3','Caption']:
  style=document.styles[name]
  assert style.font.name==WORD_FONT,(name,style.font.name)
  assert str(style.font.color.rgb).lower()==color('ink').lstrip('#').lower(),(name,str(style.font.color.rgb))
  fonts=style.element.rPr.rFonts
  assert fonts is not None
  assert not any(key.endswith('Theme') or key.endswith('cstheme') for key in fonts.attrib),name
 if table:assert abs(document.styles['Normal'].font.size.pt-WORD_TABLE_PT)<1e-8

def main():
 figures=[]
 for rel in MANIFESTS:
  path=ROOT/rel
  for item in json.loads(path.read_text())['figures']:
   assert item.get('style_profile_sha256')==STYLE_SHA256,(rel,item['stem'],'stale appearance')
   stem=path.parent/item['stem']
   svg=stem.with_suffix('.svg');root=ET.fromstring(svg.read_text())
   assert '<image' not in svg.read_text(),svg
   svg_mm=float(root.attrib['width'].removesuffix('pt'))/72*25.4
   assert abs(svg_mm-WIDTH_MM)<.05,(svg,svg_mm)
   for suffix,dpi in [('png',PNG_DPI),('tiff',TIFF_DPI),('jpg',JPG_DPI)]:
    with Image.open(stem.with_suffix('.'+suffix)) as raster:
     assert abs(raster.info['dpi'][0]-dpi)<.15,(stem,suffix,raster.info)
     assert abs(raster.width/dpi*25.4-WIDTH_MM)<.1,(stem,suffix,raster.size)
     if suffix in {'jpg','tiff'}:assert raster.mode=='RGB',(stem,suffix,raster.mode)
   figures.append({'stem':item['stem'],'manifest':rel,'style_profile_sha256':STYLE_SHA256})
 assert len(figures)==21,len(figures)
 docs=[]
 table_paths=sorted((ROOT/'02_TABLES').glob('*/*.docx'))+sorted((ROOT/'01_MANUSCRIPT/publication_tables').glob('*.docx'))
 for path in table_paths:
  style_check(Document(path),table=True);docs.append(str(path.relative_to(ROOT)))
 assert len(table_paths)==27,len(table_paths)
 for path in sorted((ROOT/'01_MANUSCRIPT').glob('*.docx')):
  style_check(Document(path));docs.append(str(path.relative_to(ROOT)))
 style_check(Document(ROOT/'00_START_HERE/OPEN_ME.docx'))
 # Best displayed values are still bold after source-table styling and book assembly.
 frame=pd.read_csv(ROOT/'02_TABLES/main/Table_2_core_comparisons.csv')
 expected=[]
 for _,group in frame.groupby(['Outcome','Model'],sort=False):
  for column,index in [('AP',3),('AUROC',4),('Brier',5)]:
   values=group[column].str.extract(r'^([0-9.]+)')[0].astype(float)
   best=values.min() if column=='Brier' else values.max()
   expected +=[(int(i)+1,index) for i in values.index[values.eq(best)]]
 assert len(expected)==19
 for path,table_index in [(ROOT/'02_TABLES/main/Table_2_core_comparisons.docx',0),(ROOT/'01_MANUSCRIPT/Main_Figures_and_Tables.docx',1)]:
  table=Document(path).tables[table_index]
  actual=[]
  for ri in range(1,len(table.rows)):
   for ci in (3,4,5):
    runs=[r for p in table.cell(ri,ci).paragraphs for r in p.runs if r.text.strip()]
    if runs and all(r.bold for r in runs):actual.append((ri,ci))
  assert set(actual)==set(expected),(path,actual,expected)
 report={'status':'passed','profile_sha256':STYLE_SHA256,'all_generated_figures':len(figures),
  'native_editable_table_documents':len(table_paths),'word_documents_checked':len(docs)+1,
  'metric_bold_cells_preserved':19,'figure_details':figures,'document_paths':docs,
  'scope':'All current and reference display generators; historical source snapshots and imported documents retain their original appearance.'}
 (ROOT/'00_START_HERE/qa/publication_style_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Shared style verified:',len(figures),'figures;',len(table_paths),'native table documents; 19 bold metric cells')
 return report
if __name__=='__main__':main()
