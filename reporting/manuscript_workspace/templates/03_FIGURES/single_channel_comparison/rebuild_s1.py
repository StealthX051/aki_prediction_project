"""Paired AP and AUROC differences; read only locally bundled frozen estimates."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '05_CODE'))
import publication_style as ps
from pathlib import Path
import json,hashlib,shutil
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image
HERE=Path(__file__).resolve().parent
DATA=HERE/'inputs/metrics_summary.csv'
FEATURES=['ecg_only','pleth_only','co2_only','awp_only','all_waveforms']
NAMES=['ECG','PLETH','CO₂','AWP','All four']
FAMILIES=['Logistic regression','XGBoost','EBM']
OUTCOMES=['any_aki','icu_admission']
ps.apply_mpl_style()

def main():
 d=pd.read_csv(DATA);d=d[d.feature_set.isin(FEATURES)].copy();assert len(d)==60
 cols=['outcome','branch','model_name','feature_set','auprc','auprc_lower','auprc_upper','auroc','auroc_lower','auroc_upper','delta_auprc','delta_auprc_lower','delta_auprc_upper','delta_auroc','delta_auroc_lower','delta_auroc_upper','n','n_patients','prevalence']
 d[cols].to_csv(HERE/'S1_absolute_and_paired_values.csv',index=False)
 manifest=[]
 for part,key,label in [('A','auprc','AP'),('B','auroc','AUROC')]:
  fig,axes=plt.subplots(2,3,figsize=(ps.WIDTH_MM/25.4,151/25.4),sharey=True)
  fig.subplots_adjust(left=.13,right=.985,top=.94,bottom=.16,wspace=.27,hspace=.43)
  for row,outcome in enumerate(OUTCOMES):
   allvalues=d[d.outcome.eq(outcome)]
   lo=100*allvalues['delta_'+key+'_lower'].min();hi=max(0,100*allvalues['delta_'+key+'_upper'].max());pad=max(1,(hi-lo)*.08)
   for col,family in enumerate(FAMILIES):
    ax=axes[row,col];subset=d[d.outcome.eq(outcome)&d.model_name.eq(family)]
    for branch,offset in [('non_windowed',-.12),('windowed',.12)]:
     encoding=ps.BRANCHES[branch];marker=encoding['marker'];face=ps.INPUT_COLORS['all_waveforms'] if encoding['filled'] else 'white'
     z=subset[subset.branch.eq(branch)].set_index('feature_set').loc[FEATURES]
     point=z['delta_'+key].to_numpy()*100;lower=z['delta_'+key+'_lower'].to_numpy()*100;upper=z['delta_'+key+'_upper'].to_numpy()*100
     ax.errorbar(point,np.arange(5)+offset,xerr=[point-lower,upper-point],fmt=marker,color=ps.INPUT_COLORS['all_waveforms'],mfc=face,mec=ps.INPUT_COLORS['all_waveforms'],ms=3.8,lw=.8,capsize=1.5)
    ax.axvline(0,color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH);ax.set(xlim=(lo-pad,hi+pad),ylim=(4.6,-.6),yticks=np.arange(5),yticklabels=NAMES)
    ax.tick_params(axis='y',labelleft=(col==0))
    ticks=MaxNLocator(3).tick_values(lo-pad,hi+pad)
    ax.set_xticks([v for v in ticks if lo-pad<=v<=hi+pad])
    ax.set_title(r'$\bf{'+chr(97+row*3+col)+'}$  '+('AKI' if row==0 else 'ICU admission')+' · '+('LR' if family=='Logistic regression' else family),loc='left',pad=6)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color=ps.color('grid'),lw=ps.GRID_WIDTH)
    if row==1:ax.set_xlabel('Δ '+label+' (percentage points)')
  fig.legend(handles=[Line2D([0],[0],marker=ps.BRANCHES['non_windowed']['marker'],color=ps.INPUT_COLORS['all_waveforms'],mfc=ps.INPUT_COLORS['all_waveforms'] if ps.BRANCHES['non_windowed']['filled'] else 'white',ls='none',label=ps.BRANCHES['non_windowed']['label']),Line2D([0],[0],marker=ps.BRANCHES['windowed']['marker'],color=ps.INPUT_COLORS['all_waveforms'],mfc=ps.INPUT_COLORS['all_waveforms'] if ps.BRANCHES['windowed']['filled'] else 'white',ls='none',label=ps.BRANCHES['windowed']['label']),Line2D([0],[0],color=ps.color('reference'),ls='--',label='Clinical-only reference')],loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False)
  stem='Figure_S1'+part+'_'+label+'_differences'
  for ext,dpi in [('svg',None),('pdf',None),('eps',None),('png',ps.PNG_DPI),('tiff',ps.TIFF_DPI),('jpg',ps.JPG_DPI)]:
   kwargs={'dpi':dpi} if dpi else {}
   if ext=='jpg':kwargs['pil_kwargs']={'quality':ps.JPEG_QUALITY,'subsampling':0}
   if ext=='tiff':kwargs['pil_kwargs']={'compression':ps.TIFF_COMPRESSION}
   fig.savefig(HERE/(stem+'.'+ext),facecolor='white',**kwargs)
   if ext=='tiff':
    with Image.open(HERE/(stem+'.tiff')) as im:rgb=im.convert('RGB')
    rgb.save(HERE/(stem+'.tiff'),dpi=(ps.TIFF_DPI,ps.TIFF_DPI),compression=ps.TIFF_COMPRESSION)
  fig.canvas.draw();r=fig.canvas.get_renderer()
  for text in fig.findobj(matplotlib.text.Text):
   if not text.get_text() or not text.get_visible():continue
   b=text.get_window_extent(r);assert b.x0>=-1 and b.x1<=fig.bbox.width+1 and b.y0>=-1 and b.y1<=fig.bbox.height+1,(stem,text.get_text())
  plt.close(fig)
  metric_name = "average precision (AP)" if key == "auprc" else "AUROC"
  caption = 'Waveform-only minus same-family clinical-only {metric}, in percentage points. Filled circles show whole-case models; open squares show windowed models. Whiskers are 95% paired patient-cluster bootstrap confidence intervals. AWP, airway pressure; PLETH, plethysmography.\n'.format(metric=metric_name)
  (HERE/(stem+'.caption.txt')).write_text(caption)
  manifest.append({'style_profile_sha256':ps.STYLE_SHA256,'figure':'S1'+part,'stem':stem,'title':'Waveform-only '+label+' differences','width_mm':ps.WIDTH_MM,'height_mm':151,'metric':key,'comparisons':60,'caption_file':stem+'.caption.txt'})
 (HERE/'figure_manifest.json').write_text(json.dumps({'figures':manifest,'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'recomputed_bootstrap':False,'paired_estimates_retained':120},indent=2)+'\n')
 print('S1A/S1B rendered:120 paired estimates and intervals; source values unchanged.')

if __name__=='__main__':main()
