"""Captioned A4 vector PDF proofs; uses only portable local figure masters."""
from pathlib import Path
from publication_navigation import add_pdf_navigation, pdf_pages
from publication_style import FONT_FAMILY, WIDTH_MM, CAPTION_PT, STYLE, STYLE_SHA256, style_context
import json,re,subprocess,importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('documents',Path(__file__).with_name('build_manuscript_documents.py'))
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
OUT=ROOT/'01_MANUSCRIPT';PROOFS=OUT/'figure_proofs';PROOFS.mkdir(exist_ok=True)
MM=72/25.4;W,H=210*MM,297*MM
MEASURE=TextToPath()
PROOF=STYLE['proof']

def width(text,size=CAPTION_PT,bold=False):
 return MEASURE.get_text_width_height_descent(text,FontProperties(family=FONT_FAMILY,size=size,weight='bold' if bold else 'normal'),False)[0]

def wrap(text,available,size=CAPTION_PT,first=None,bold=False):
 lines=[];line=''
 for word in text.split():
  candidate=(line+' '+word).strip();limit=(first if first is not None and not lines else available)
  if line and width(candidate,size,bold)>limit:lines.append(line);line=word
  else:line=candidate
 if line:lines.append(line)
 return lines

def fragment(path,x=0,y=0,frame=False):
 content=path.read_text();prefix='PageFrame_' if frame else 'PageArt_'
 names=set(re.findall(r'/FontName /([^\s]+)',content))
 for name in sorted(names,key=len,reverse=True):
  content=re.sub(r'/'+re.escape(name)+r'(?![\w-])','/'+prefix+name,content)
 return '\ngsave\n'+f'{x:.8f} {y:.8f} translate\n%%BeginDocument: '+path.name+'\n'+content+'\n%%EndDocument\ngrestore\n'

def methods_pdf():
 target=OUT/'Supplementary_Methods.pdf';pages=0
 with style_context(),PdfPages(target) as pdf:
  fig=None;y=0
  def page():
   nonlocal fig,y,pages
   if fig is not None:pdf.savefig(fig);plt.close(fig)
   fig=plt.figure(figsize=(210/25.4,297/25.4));y=H-17*MM;pages+=1
   fig.text(18*MM/W,y/H,'Supplementary methods'+(' (continued)' if pages>1 else ''),fontsize=PROOF['title_pt'],weight='bold',va='top');y-=27
  page()
  for i,(heading,text) in enumerate(D.method_blocks()):
   if i==3:page()
   lines=wrap(text,WIDTH_MM*MM,PROOF['body_pt']);heading_lines=wrap(heading,WIDTH_MM*MM,PROOF['heading_pt'],bold=True)
   if y-(len(heading_lines)*14+36)<16*MM:page()
   fig.text(18*MM/W,y/H,'\n'.join(heading_lines),fontsize=PROOF['heading_pt'],weight='bold',va='top',linespacing=1.2);y-=len(heading_lines)*14+5
   for line in lines:
    if y-13.5<16*MM:page()
    fig.text(18*MM/W,y/H,line,fontsize=PROOF['body_pt'],va='top');y-=13.5
   y-=10
  pdf.savefig(fig);plt.close(fig)
 return pages

def build():
 main,supp,_=D.entries();checks=[]
 active={f['stem'] for f in main+supp}
 for old in PROOFS.glob('*.pdf'):
  if old.stem not in active:old.unlink()
 for item in main+supp:
  stem=item['stem'];label=item['label'].replace('Figure','Fig.');text=D.caption_text(item).rstrip('.')
  prefix=label+' ';prefix_w=width(prefix,CAPTION_PT,True)
  lines=wrap(text,WIDTH_MM*MM,CAPTION_PT,first=WIDTH_MM*MM-prefix_w)
  art_h=item['height_mm']*MM;image_y=H-13*MM-art_h;caption_top=image_y-5*MM;bottom=caption_top-len(lines)*11.5
  assert bottom>16*MM,(label,bottom/MM)
  with style_context():
   fig=plt.figure(figsize=(210/25.4,297/25.4));fig.patch.set_facecolor('white')
   fig.text(18*MM/W,caption_top/H,prefix,fontsize=CAPTION_PT,weight='bold',va='top')
   for i,line in enumerate(lines):fig.text((18*MM+(prefix_w if i==0 else 0))/W,(caption_top-i*11.5)/H,line,fontsize=CAPTION_PT,va='top')
   frame=PROOFS/(stem+'.frame.eps');fig.savefig(frame,format='eps');plt.close(fig)
  ps=PROOFS/(stem+'.ps');dest=PROOFS/(stem+'.pdf')
  content='%!PS-Adobe-3.0\n%%Pages: 1\n%%Page: 1 1\n'+f'<< /PageSize [{W:.8f} {H:.8f}] >> setpagedevice\n'+'/pageSave save def\n/showpage {} def\n'
  content+=fragment(frame,frame=True)+fragment(ROOT/item['folder']/(stem+'.eps'),18*MM,image_y)
  ps.write_text(content+'\npageSave restore\nshowpage\n%%EOF\n')
  subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=pdfwrite','-dCompatibilityLevel=1.7','-sOutputFile='+str(dest),str(ps)],check=True)
  frame.unlink();ps.unlink()
  checks.append({'label':label,'file':str(dest.relative_to(ROOT)),'artwork_scale':1,'width_mm':WIDTH_MM,'caption_bottom_mm':bottom/MM,'caption_lines':len(lines)})
 method_pages=methods_pdf()
 for name,inputs in [('Main_Figures_and_Tables',[PROOFS/(main[0]['stem']+'.pdf'),ROOT/'02_TABLES/main/Table_1_cohort.pdf',ROOT/'02_TABLES/main/Table_2_core_comparisons.pdf']+[PROOFS/(f['stem']+'.pdf') for f in main[1:]]),('Supplementary_Figures_and_Methods',[OUT/'Supplementary_Methods.pdf']+[PROOFS/(f['stem']+'.pdf') for f in supp])]:
  target=OUT/(name+'.pdf')
  subprocess.run(['pdfunite',*[str(p) for p in inputs],str(target)],check=True)
  if name=='Main_Figures_and_Tables':
   headings=[main[0]['label']+' — '+main[0]['title'],'Table 1 — Cohort characteristics','Table 2 — Whole-case model performance']+[f['label']+' — '+f['title'] for f in main[1:]]
  else:headings=['Supplementary methods']+[f['label']+' — '+f['title'] for f in supp]
  page=1;sections=[]
  for heading,source in zip(headings,inputs):sections.append((heading,page));page+=pdf_pages(source)
  navigation=add_pdf_navigation(target,name.replace('_',' '),sections)
  target.with_suffix('.navigation.json').write_text(json.dumps(navigation,indent=2)+'\n')
 (ROOT/'00_START_HERE/qa/publication_proofs.json').write_text(json.dumps({'figure_pages':checks,'supplementary_methods_pages':method_pages,'style_profile_sha256':STYLE_SHA256,'vector_master_preserved':True,'proof_is_native_vector_composition_not_a_Word_render':True},indent=2)+'\n')
 print('Built',len(checks),'captioned native-vector pages and',method_pages,'methods pages')
if __name__=='__main__':build()
