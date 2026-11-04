"""Rebuild the held-out EBM illustration from portable contribution CSVs only."""

import publication_style as ps
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'03_FIGURES/heldout_cases'
STEM='heldout_ebm_cases'
CAPTION='Held-out EBM explanations for cases nearest the 10th and 90th percentiles of combined-model AKI or ICU-admission risk. Selection did not use observed outcomes. Filled and open bars raise and lower calibrated log-odds; gray and green bars show the calibrated model intercept and final prediction. Clinical contributions are grouped by feature; waveform contributions by channel. Headers give clinical-only and combined risks. WBC, white blood cell count (10³/µL); BUN, blood urea nitrogen (mg/dL); Hb, hemoglobin (g/dL); ASA, American Society of Anesthesiologists physical status.'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 cases=pd.read_csv(OUT/'case_selection_ledger.csv')
 groups=pd.read_csv(OUT/'grouped_calibrated_contributions.csv')
 context=pd.read_csv(OUT/'case_clinical_context.csv').set_index(['display_case','feature'])
 assert len(cases)==4 and len(groups)==32
 assert not cases.selection_consulted_outcome.any()
 for row in cases.itertuples():
  data=groups[groups.display_case.eq(row.display_case)].sort_values('order')
  assert abs(row.calibrated_baseline+data.contribution.sum()-row.calibrated_final_logit)<1e-12
  assert abs(1/(1+np.exp(-row.calibrated_final_logit))-row.combined_probability)<1e-12
 values=np.r_[groups.start.to_numpy(),groups.end.to_numpy(),cases.calibrated_baseline.to_numpy(),cases.calibrated_final_logit.to_numpy(),0]
 low,high=values.min(),values.max();span=high-low;xlim=(low-.10*span,high+.10*span)
 with ps.style_context(**{'axes.spines.top': False, 'axes.spines.right': False}):
  fig=plt.figure(figsize=(ps.WIDTH_MM/25.4,215/25.4));grid=fig.add_gridspec(2,2,left=.155,right=.985,top=.890,bottom=.155,wspace=.62,hspace=.56)
  axes=[]
  for i,row in enumerate(cases.itertuples()):
   ax=fig.add_subplot(grid[i//2,i%2]);axes.append(ax)
   data=groups[groups.display_case.eq(row.display_case)].sort_values('order')
   short={'White blood cell count':'WBC','Blood urea nitrogen':'BUN','ASA physical status':'ASA class','Surgical department':'Department','Other clinical terms':'Other clinical'}
   clinical_labels=[]
   for item in data.itertuples():
    if item.order>3:clinical_labels.append(short.get(item.label,item.label));continue
    value=context.loc[(row.display_case,item.group),'raw_value']
    if item.group=='preop_wbc':label='WBC: '+('missing' if pd.isna(value) else f'{float(value):.2f}')
    elif item.group=='preop_hb':label=f'Hb: {float(value):.1f}'
    elif item.group=='preop_bun':label=f'BUN: {float(value):.0f}'
    elif item.group=='age':label=f'Age: {float(value):.0f} years'
    elif item.group=='asa':label='ASA: '+{1:'I',2:'II',3:'III',4:'IV'}.get(int(float(value)),str(value))
    elif item.group=='optype':label='Procedure:\n'+str(value)
    elif item.group=='department':label='Department:\n'+str(value)
    else:label=short.get(item.label,item.label)
    clinical_labels.append(label)
   labels=['Intercept']+clinical_labels+['Prediction'];baseline=row.calibrated_baseline;final=row.calibrated_final_logit
   positions=[0,1.4,2.8,4.2,5.4,6.4,7.4,8.4,9.4,10.5]
   ax.barh(0,baseline,left=0,height=.64,facecolor=ps.color('baseline'),edgecolor=ps.color('baseline_edge'),linewidth=.65,zorder=3)
   for j,item in enumerate(data.itertuples(),1):
    start,end=item.start,item.end
    ax.plot([start,start],[positions[j-1]+.34,positions[j]-.34],color=ps.color('connector'),lw=.65,zorder=2)
    ax.barh(positions[j],abs(end-start),left=min(start,end),height=.64,facecolor=ps.color('signed_contribution') if end>=start else 'white',edgecolor=ps.color('signed_contribution'),linewidth=.85,zorder=3)
   ax.plot([final,final],[positions[-2]+.34,positions[-1]-.34],color=ps.color('connector'),lw=.65,zorder=2)
   ax.barh(positions[-1],final,left=0,height=.64,facecolor=ps.INPUT_COLORS['preop_and_all_waveforms'],edgecolor=ps.color('combined_edge'),linewidth=.65,zorder=3)
   ax.set(yticks=positions,yticklabels=labels,ylim=(positions[-1]+.55,-.65),xlim=xlim)
   ticks=MaxNLocator(4).tick_values(*xlim);ax.set_xticks([tick for tick in ticks if xlim[0]<=tick<=xlim[1]]);ax.tick_params(axis='y',length=0,pad=3)
   ax.axvline(0,color=ps.color('zero'),lw=ps.REFERENCE_WIDTH,zorder=1);ax.grid(axis='x',color=ps.color('grid'),linewidth=ps.GRID_WIDTH,zorder=0)
   ax.set_xlabel('Calibrated log-odds',labelpad=4)
   title=r'$\bf{'+chr(97+i)+'}$  '+f'{row.outcome_label} · {int(row.target_quantile*100)}th percentile'
   ax.text(0,1.20,title,transform=ax.transAxes,fontsize=ps.STYLE['figure']['title_pt'],fontweight='normal',va='bottom')
   summary=f'Clinical {row.clinical_probability*100:.1f}%  |  Combined {row.combined_probability*100:.1f}%'
   status='Yes' if row.observed_outcome else 'No'
   ax.text(0,1.07,summary+'\nOutcome: '+status,transform=ax.transAxes,fontsize=ps.STYLE['figure']['body_pt'],va='bottom',linespacing=1.25)
   # Terminal probability is placed outside the plotting box in the panel key,
   # rather than mislabeled as an additive probability contribution.
  fig.legend(handles=[Patch(facecolor=ps.color('signed_contribution'),edgecolor=ps.color('signed_contribution'),label='Raises risk'),Patch(facecolor='white',edgecolor=ps.color('signed_contribution'),label='Lowers risk'),Patch(facecolor=ps.color('baseline'),edgecolor=ps.color('baseline_edge'),label='Intercept'),Patch(facecolor=ps.INPUT_COLORS['preop_and_all_waveforms'],edgecolor=ps.color('combined_edge'),label='Prediction')],loc='lower center',bbox_to_anchor=(.5,.010),ncol=2,frameon=False,handlelength=1.5,columnspacing=2.0)
  fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
  for text in fig.findobj(matplotlib.text.Text):
   if not text.get_visible() or not text.get_text().strip():continue
   bbox=text.get_window_extent(renderer)
   if bbox.x0<-.5 or bbox.y0<-.5 or bbox.x1>fig.bbox.width+.5 or bbox.y1>fig.bbox.height+.5:outside.append(text.get_text())
  assert not outside,('off-canvas text',outside)
  for extension in ['svg','pdf','eps']:fig.savefig(OUT/(STEM+'.'+extension),format=extension)
  fig.savefig(OUT/(STEM+'.png'),dpi=ps.PNG_DPI)
  png=Image.open(OUT/(STEM+'.png')).convert('RGB')
  fig.savefig(OUT/(STEM+'.tiff'),dpi=ps.TIFF_DPI,pil_kwargs={'compression':ps.TIFF_COMPRESSION})
  with Image.open(OUT/(STEM+'.tiff')) as raster:tiff=raster.convert('RGB')
  tiff.save(OUT/(STEM+'.tiff'),compression=ps.TIFF_COMPRESSION,dpi=(ps.TIFF_DPI,ps.TIFF_DPI))
  jpg=png.resize((round(ps.WIDTH_MM/25.4*ps.JPG_DPI),round(215/25.4*ps.JPG_DPI)),Image.Resampling.LANCZOS);jpg.save(OUT/(STEM+'.jpg'),quality=ps.JPEG_QUALITY,dpi=(ps.JPG_DPI,ps.JPG_DPI))
  plt.close(fig)
 (OUT/(STEM+'.caption.txt')).write_text(CAPTION+'\n')
 groups.to_csv(OUT/(STEM+'.plotted.csv'),index=False)
 checks={'status':'passed','case_count':4,'contribution_groups':len(groups),'prediction_reconstruction_tolerance':1e-12,'off_canvas_labels':0,'shared_axis_limits_calibrated_logit':list(xlim),'svg_has_no_raster_images':'<image' not in (OUT/(STEM+'.svg')).read_text(),'displayed_case_identifiers':'Case A–D only; original IDs retained in selection ledger','data_sha256':{'case_selection_ledger.csv':sha(OUT/'case_selection_ledger.csv'),'case_clinical_context.csv':sha(OUT/'case_clinical_context.csv'),'grouped_calibrated_contributions.csv':sha(OUT/'grouped_calibrated_contributions.csv')},'models_fitted':0}
 (OUT/'figure_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
 item={'style_profile_sha256':ps.STYLE_SHA256,'stem':STEM,'title':'EBM case explanations','label':'Figure S10','width_mm':ps.WIDTH_MM,'height_mm':215,'caption_file':STEM+'.caption.txt','plotted_data_file':STEM+'.plotted.csv','figure_role':'supplementary illustration candidate','formats':['svg','pdf','eps','png','tiff','jpg'],'font_range_pt':[ps.STYLE['figure']['body_pt'],ps.STYLE['figure']['title_pt']]}
 (OUT/'figure_manifest.json').write_text(json.dumps({'figures':[item],'portable_rebuild':'05_CODE/rebuild_case_figure.py','checks':checks},indent=2)+'\n')
 print('Held-out case figure built and exact reconstruction checks passed')
if __name__=='__main__':main()
