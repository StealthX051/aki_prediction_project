"""Validate portable DCA source replay, vectors, scales, fonts and PDF text bounds."""
from pathlib import Path
import hashlib, importlib.util, json, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent

def validate(relocate=False):
    spec=importlib.util.spec_from_file_location('portable_dca',HERE/'rebuild_dca.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    groups=[];source_rows=0;paired_rows=0
    for scope in mod.OUTCOMES:
      for branch in ('non_windowed','windowed'):
        core,delta,checks=mod.load(scope,branch)
        groups.extend(checks);source_rows+=len(core);paired_rows+=len(delta)
    manifest=json.loads((HERE/'figures/dca_manifest.json').read_text())
    formats=[];pages=0;off_page=[]
    for figure in manifest['figures']:
      stem=figure['stem'];pdf=HERE/'figures'/(stem+'.pdf')
      assert '<image' not in (HERE/'figures'/(stem+'.svg')).read_text()
      assert figure['width_mm']==174 and 8<=figure['font_range_pt'][0]<=figure['font_range_pt'][1]<=12
      if figure['view']=='absolute_detail':
        assert all(v['model_points_below']==0 for v in figure['detail_clipping'])
        assert all(v['explicit_down_arrow'] for v in figure['detail_clipping'] if v['treat_all_points_below_detail_axis'])
      if figure['view']=='absolute_full_range':
        assert all(v['treat_all_points_below_detail_axis']==v['model_points_below']==0 for v in figure['detail_clipping'])
      for ext,dpi in [('png',600),('tiff',600),('jpg',300)]:
        with Image.open(HERE/'figures'/(stem+'.'+ext)) as im:
          assert abs(float(im.info['dpi'][0])-dpi)<.1
          assert abs(im.width/dpi*25.4-174)<.05
          if ext in ('jpg','tiff'):assert im.mode=='RGB'
          formats.append({'stem':stem,'format':ext,'dpi':float(im.info['dpi'][0]),'mode':im.mode})
      text=subprocess.check_output(['pdftotext','-bbox',str(pdf),'-'],text=True)
      tree=ET.fromstring(text)
      for page in tree.iter('{http://www.w3.org/1999/xhtml}page'):
        pages+=1;w=float(page.attrib['width']);h=float(page.attrib['height'])
        for word in page.iter('{http://www.w3.org/1999/xhtml}word'):
          if float(word.attrib['xMin'])<-.5 or float(word.attrib['yMin'])<-.5 or float(word.attrib['xMax'])>w+.5 or float(word.attrib['yMax'])>h+.5:
            off_page.append({'stem':stem,'text':word.text})
    assert not off_page,off_page
    sources=json.loads((HERE/'source_provenance.json').read_text())['files']
    for file in sources:assert hashlib.sha256((HERE/file['portable_copy']).read_bytes()).hexdigest()==file['sha256']
    code=(HERE/'rebuild_dca.py').read_text()
    assert '/media/' not in code and '/home/' not in code
    portability={'relative_path_source_access':True,'rebuild_has_no_VM_or_project_path':True,'relocated_rebuild_tested':False}
    if relocate:
      clone=Path(tempfile.mkdtemp(prefix='vitaldb-dca-portable-'))
      shutil.copy2(HERE/'rebuild_dca.py',clone/'rebuild_dca.py');shutil.copytree(HERE/'data',clone/'data')
      provenance=json.loads((HERE/'source_provenance.json').read_text())
      for record in provenance['files']:record['source_original']='/unavailable-original-VM/'+Path(record['source_original']).name
      (clone/'source_provenance.json').write_text(json.dumps(provenance))
      result=subprocess.run([sys.executable,str(clone/'rebuild_dca.py'),'--scope','icu_waveform_sensitivity','--branch','windowed'],cwd='/tmp',text=True,capture_output=True)
      assert result.returncode==0,result.stdout+result.stderr
      portability.update({'relocated_rebuild_tested':True,'original_source_paths_deliberately_unavailable':True,'scope':'Broader ICU','branch':'Windowed','returncode':result.returncode,'output':result.stdout.strip()})
      shutil.rmtree(clone)
    result={'complete_core_source_rows_checked':source_rows,'paired_source_rows_checked':paired_rows,'cohort_branch_family_checks':len(groups),
            'numeric_checks':groups,'source_copies_bytewise_equal':True,'figures_checked':len(manifest['figures']),
            'formats_checked':formats,'PDF_pages_checked':pages,'off_page_words':off_page,'portability':portability,
            'fitting_or_bootstrap_performed':False}
    (HERE/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['complete_core_source_rows_checked','paired_source_rows_checked','cohort_branch_family_checks','figures_checked','PDF_pages_checked','portability']},indent=2))

if __name__=='__main__':validate(relocate='--relocate' in sys.argv)
