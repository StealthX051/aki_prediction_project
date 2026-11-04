"""Portable supporting ROC curves from frozen held-out probabilities; no fitted estimates changed."""

import publication_style as ps
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'03_FIGURES/roc_curves';OUT.mkdir(exist_ok=True)
STEM='S2_whole_case_ROC'
FAMILIES=[('logistic_regression','LR'),('xgboost','XGBoost'),('ebm','EBM')]
OUTCOMES=[('any_aki','AKI'),('icu_admission','ICU admission')]
INPUTS=[(key,ps.INPUT_LABELS[key],ps.INPUT_COLORS[key],ps.INPUT_STYLES[key]) for key in ps.INPUT_ORDER]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def export_data():
 index=pd.read_csv(ROOT/'04_DATA/prediction_index.csv')
 metrics=pd.read_csv(ROOT/'04_DATA/results/main_shared/tables/metrics_summary.csv')
 curves=[];sources=[]
 for outcome,_ in OUTCOMES:
  for family,name in FAMILIES:
   for inputs,_,_,_ in INPUTS:
    record=index[index.cohort_scope.eq('main_shared')&index.branch.eq('non_windowed')&index.model.eq(family)&index.outcome.eq(outcome)&index.feature_set.eq(inputs)]
    assert len(record)==1,(outcome,family,inputs)
    path=ROOT/record.iloc[0].offline_prediction_path;d=pd.read_csv(path)
    assert len(d)==3571 and d.caseid.nunique()==3571
    score=d.y_prob_calibrated.to_numpy();truth=d.y_true.to_numpy();assert np.isfinite(score).all()
    order=np.argsort(-score,kind='stable');scores=score[order];ys=truth[order]
    endpoints=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1]
    true_positive=np.r_[0,np.cumsum(ys)[endpoints]];false_positive=np.r_[0,(1+endpoints)-np.cumsum(ys)[endpoints]]
    tpr=true_positive/ys.sum();fpr=false_positive/(len(ys)-ys.sum());auc=float(np.trapezoid(tpr,fpr))
    expected=metrics[metrics.outcome.eq(outcome)&metrics.branch.eq('non_windowed')&metrics.model_name.eq({'LR':'Logistic regression','XGBoost':'XGBoost','EBM':'EBM'}[name])&metrics.feature_set.eq(inputs)]
    assert len(expected)==1 and abs(auc-expected.iloc[0].auroc)<1e-12,(outcome,family,inputs,auc)
    for point,(x,y) in enumerate(zip(fpr,tpr)):curves.append({'outcome':outcome,'family':family,'feature_set':inputs,'point':point,'fpr':x,'tpr':y,'auroc':auc})
    sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'operations':len(d),'events':int(ys.sum()),'auroc_matches_frozen_summary':True})
 pd.DataFrame(curves).to_csv(OUT/(STEM+'.plotted.csv'),index=False)
 (OUT/'source_audit.json').write_text(json.dumps({'sources':sources,'metrics_summary_sha256':sha(ROOT/'04_DATA/results/main_shared/tables/metrics_summary.csv'),'model_fits':0,'bootstrap_draws':0},indent=2)+'\n')

def build():
 data=pd.read_csv(OUT/(STEM+'.plotted.csv'))
 with ps.style_context():
  fig,axes=plt.subplots(2,3,figsize=(ps.WIDTH_MM/25.4,151/25.4),sharex=True,sharey=True)
  fig.subplots_adjust(left=.085,right=.96,bottom=.135,top=.94,wspace=.15,hspace=.36)
  for ri,(outcome,olabel) in enumerate(OUTCOMES):
   for ci,(family,flabel) in enumerate(FAMILIES):
    ax=axes[ri,ci];ax.set_title(r'$\bf{'+chr(97+ri*3+ci)+'}$  '+olabel+' · '+flabel,loc='left',fontsize=ps.STYLE['figure']['title_pt'])
    values=[]
    for inputs,label,color,style in INPUTS:
     s=data[data.outcome.eq(outcome)&data.family.eq(family)&data.feature_set.eq(inputs)].sort_values('point')
     assert not s.empty
     ax.plot(s.fpr,s.tpr,color=color,ls=style,lw=ps.CURVE_WIDTH);values.append(f'{s.auroc.iloc[0]:.3f}')
    ax.plot([0,1],[0,1],color=ps.color('reference'),ls=ps.REFERENCE_STYLE,lw=ps.REFERENCE_WIDTH,zorder=0)
    ax.text(.98,.05,'AUROC\n'+' / '.join(values),transform=ax.transAxes,ha='right',va='bottom',fontsize=ps.STYLE['figure']['body_pt'])
    ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1]);ax.grid(color=ps.color('grid'),lw=ps.GRID_WIDTH)
    ax.spines[['top','right']].set_visible(False)
    if ri==1:ax.set_xlabel('False-positive rate')
    if ci==0:ax.set_ylabel('Sensitivity')
  handles=[Line2D([0],[0],color=c,ls=style,lw=ps.CURVE_WIDTH,label=label) for _,label,c,style in INPUTS]
  fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False,handlelength=2)
  fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
  for text in fig.findobj(matplotlib.text.Text):
   if not text.get_visible() or not text.get_text():continue
   bb=text.get_window_extent(renderer)
   # Shared-axis hidden ticks may have empty extents; visible text must remain on canvas.
   if bb.width and bb.height:assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1,(text.get_text(),list(bb.bounds))
   bounds.append(text.get_text())
  for ext in ['svg','pdf','eps']:fig.savefig(OUT/(STEM+'.'+ext),format=ext)
  fig.savefig(OUT/(STEM+'.png'),dpi=ps.PNG_DPI,facecolor='white')
  with Image.open(OUT/(STEM+'.png')) as im:
   im.convert('RGB').save(OUT/(STEM+'.tiff'),compression=ps.TIFF_COMPRESSION,dpi=(ps.TIFF_DPI,ps.TIFF_DPI))
  fig.savefig(OUT/(STEM+'.jpg'),dpi=300,facecolor='white',pil_kwargs={'quality':ps.JPEG_QUALITY})
  plt.close(fig)
 caption='Whole-case ROC curves from pooled outer-test predictions. Rows show AKI and ICU admission; columns show LR, XGBoost and EBM. AUROCs follow the legend order: clinical-only, waveform-only and combined. Gray diagonals mark chance discrimination.'
 (OUT/(STEM+'.caption.txt')).write_text(caption+'\n')
 manifest={'figures':[{'style_profile_sha256':ps.STYLE_SHA256,'stem':STEM,'label':'Figure S2','title':'ROC curves','width_mm':ps.WIDTH_MM,'height_mm':151,'caption_file':STEM+'.caption.txt','plotted_data_file':STEM+'.plotted.csv','font_range_pt':[8,9],'auroc_matches_frozen_summary':True,'visible_text_bounds_pass':True,'vector_svg_has_no_raster':'<image' not in (OUT/(STEM+'.svg')).read_text()}]}
 (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('ROC: 18 curves, saved AUROCs matched; six formats and caption built')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--export-source',action='store_true');args=parser.parse_args()
 if args.export_source:export_data()
 build()
