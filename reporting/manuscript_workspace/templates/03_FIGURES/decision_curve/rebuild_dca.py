"""Rebuild conventional and paired DCA from local saved aggregates only.

No VM paths, model fitting, prediction generation, calibration, threshold selection
or bootstrap calculations are required. Run from any working directory.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '05_CODE'))
import publication_style as ps
from pathlib import Path
import argparse, hashlib, json, platform, re
from importlib.metadata import version
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
MM=1/25.4
FAMILIES=[('logistic_regression','LR'),('xgboost','XGBoost'),('ebm','EBM')]
OUTCOMES={'main_shared':[('any_aki','AKI'),('icu_admission','ICU admission')],
          'icu_waveform_sensitivity':[('icu_admission','ICU admission')],
          'mortality_waveform_sensitivity':[('mortality','Mortality')]}
SCOPE_NAMES={'main_shared':'Main shared cohort','icu_waveform_sensitivity':'Broader ICU sensitivity cohort','mortality_waveform_sensitivity':'Exploratory mortality cohort'}
BRANCH_NAMES={'non_windowed':'Whole case','windowed':'Windowed'}
READER_TITLES = {'S5A_dca_absolute_detail': 'Decision curves', 'S5A_dca_absolute_full_range': 'Decision curves, full range', 'DCA_all_inputs_full_range': 'Decision curves, all inputs', 'S5B_dca_paired_difference': 'Net-benefit differences'}
INPUTS=['preop_only','all_waveforms','preop_and_all_waveforms']
LABELS=ps.INPUT_LABELS
COLORS=ps.INPUT_COLORS
STYLES=ps.INPUT_STYLES
DETAIL_LIMITS={'any_aki':(-.5,5.5),'icu_admission':(-2.5,24),'mortality':(-.3,1.15)}
XTICKS={'any_aki':[1,10,20,30],'icu_admission':[5,20,40,60],'mortality':[.1,1,3,5]}
CONVENTIONAL_NOTE='Whole-case clinical-only and combined decision curves with treat-all and treat-none references. Net benefit is per 100 operations. Arrows mark the treat-all curve below the plotting range.'
DELTA_NOTE='Combined-minus-clinical net benefit within model families, per 100 operations. Lines and shading show estimates and pointwise 95% paired patient-cluster bootstrap confidence intervals. Dashed lines mark zero.'
FULL_NOTE='Whole-case clinical-only and combined decision curves with treat-all and treat-none references. Net benefit per 100 operations is shown over the full vertical range.'
ALL_INPUTS_NOTE='Whole-case clinical-only, waveform-only and combined decision curves with treat-all and treat-none references. Net benefit per 100 operations is shown over the full vertical range.'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load(scope,branch):
    summary=pd.read_csv(DATA/f'{scope}__decision_curve_summary.csv')
    paired=pd.read_csv(DATA/f'{scope}__decision_curve_comparisons.csv')
    metrics=pd.read_csv(DATA/f'{scope}__metrics_summary.csv')
    core=summary[summary.branch.eq(branch)&summary.feature_set.isin(INPUTS)].copy()
    delta=paired[paired.branch.eq(branch)&paired.reference_feature_set.eq('preop_only')&
                 paired.candidate_feature_set.eq('preop_and_all_waveforms')&
                 paired.reference_model.eq(paired.candidate_model)].copy()
    checks=[]
    for outcome,_ in OUTCOMES[scope]:
      outcome_core=core[core.outcome.eq(outcome)]
      expected=np.sort(outcome_core.threshold_probability.unique())
      assert len(expected)>1
      for family,_ in FAMILIES:
        g=outcome_core[outcome_core.model_type.eq(family)]
        models={name:g[g.feature_set.eq(name)].sort_values('threshold_probability') for name in INPUTS}
        for name,rows in models.items():
          assert len(rows)==len(expected) and not rows.threshold_probability.duplicated().any()
          np.testing.assert_allclose(rows.threshold_probability,expected,rtol=0,atol=0)
          assert rows[['model','treat_all','treat_none','model_lower','model_upper']].notna().all().all()
          assert rows.n_operations.nunique()==rows.n_patients.nunique()==1
        r=metrics[metrics.outcome.eq(outcome)&metrics.branch.eq(branch)&metrics.feature_set.eq('preop_only')&metrics.model_name.eq({'logistic_regression':'Logistic regression','xgboost':'XGBoost','ebm':'EBM'}[family])]
        assert len(r)==1
        prevalence=float(r.prevalence.iloc[0])
        reference=models['preop_only']
        odds=expected/(1-expected)
        np.testing.assert_allclose(reference.treat_all,prevalence-(1-prevalence)*odds,rtol=0,atol=1e-12)
        assert (reference.treat_none==0).all()
        for rows in models.values():
          np.testing.assert_allclose(rows[['treat_all','treat_none']],reference[['treat_all','treat_none']],rtol=0,atol=1e-12)
          assert set(rows.bootstrap_draw_hash)==set(reference.bootstrap_draw_hash)
        d=delta[delta.outcome.eq(outcome)&delta.candidate_model.eq(family)].sort_values('threshold_probability')
        assert len(d)==len(expected) and d.comparison_id.nunique()==1
        np.testing.assert_allclose(d.threshold_probability,expected,rtol=0,atol=0)
        np.testing.assert_allclose(d.delta,models['preop_and_all_waveforms'].model.to_numpy()-reference.model.to_numpy(),rtol=0,atol=1e-12)
        assert set(d.bootstrap_draw_hash)==set(reference.bootstrap_draw_hash)
        checks.append({'cohort_scope':scope,'branch':branch,'outcome':outcome,'family':family,'threshold_points':len(expected),'grid_min':float(expected[0]),'grid_max':float(expected[-1]),
          'prevalence':prevalence,'operations':int(reference.n_operations.iloc[0]),'patients':int(reference.n_patients.iloc[0]),
          'treat_none_zero':True,'treat_all_formula_verified':True,'paired_difference_verified':True,'shared_draw_identity_verified':True})
    return core,delta,checks


def style_axes(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color=ps.color('grid'),lw=ps.GRID_WIDTH)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5,width=.5)


def canvas(rows):
    height=158 if rows==2 else 98
    fig,axes=plt.subplots(rows,3,figsize=(ps.WIDTH_MM*MM,height*MM),squeeze=False)
    fig.subplots_adjust(left=.12,right=.985,bottom=.18 if rows==2 else .25,top=.94,hspace=.48,wspace=.30)
    return fig,axes


def title(ax,letter,label):
    ax.set_title(r'$\bf{'+letter+r'}$  '+label,loc='left',pad=6)


def legend(fig,inputs,delta=False,detail=False):
    if delta:
      handles=[Line2D([0],[0],color=COLORS['preop_and_all_waveforms'],lw=ps.CURVE_WIDTH,label='Combined − clinical'),
               Line2D([0],[0],color=ps.color('reference'),lw=ps.REFERENCE_WIDTH,ls='--',label='No difference')]
      cols=2
    else:
      handles=[Line2D([0],[0],color=COLORS[k],ls=STYLES[k],lw=ps.CURVE_WIDTH,label=LABELS[k]) for k in inputs]
      handles.extend([Line2D([0],[0],color=ps.color('reference'),ls='-.',lw=ps.REFERENCE_WIDTH,label='Treat all'),
                      Line2D([0],[0],color=ps.color('ink'),ls='-',lw=ps.REFERENCE_WIDTH,label='Treat none')])
      cols=2 if len(handles)==4 else 3
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.012),ncol=cols,frameon=False,handlelength=2.8,columnspacing=1.5)


def save(fig,stem,out,caption,metadata):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    w,h=fig.get_size_inches()*25.4
    assert abs(w-ps.WIDTH_MM)<1e-8
    fonts=[];bounds=[]
    for t in fig.findobj(matplotlib.text.Text):
      if not t.get_visible() or not t.get_text():continue
      fonts.append(t.get_fontsize())
      b=t.get_window_extent(renderer)
      if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:bounds.append(t.get_text())
    assert not bounds,(stem,bounds)
    assert min(fonts)>=8 and max(fonts)<=12
    out.mkdir(parents=True,exist_ok=True)
    for ext,dpi in [('svg',None),('pdf',None),('eps',None),('png',ps.PNG_DPI),('tiff',ps.TIFF_DPI),('jpg',ps.JPG_DPI)]:
      args={'dpi':dpi} if dpi else {}
      if ext=='jpg':args['pil_kwargs']={'quality':ps.JPEG_QUALITY,'subsampling':0}
      if ext=='tiff':args['pil_kwargs']={'compression':ps.TIFF_COMPRESSION}
      fig.savefig(out/f'{stem}.{ext}',**args)
      if ext=='tiff':
        with Image.open(out/f'{stem}.{ext}') as im:rgb=im.convert('RGB')
        rgb.save(out/f'{stem}.{ext}',compression=ps.TIFF_COMPRESSION,dpi=(ps.TIFF_DPI,ps.TIFF_DPI))
    assert '<image' not in (out/f'{stem}.svg').read_text()
    (out/f'{stem}.caption.txt').write_text(caption+'\n')
    metadata.update({'style_profile_sha256':ps.STYLE_SHA256,'stem':stem,'title':READER_TITLES[stem],'width_mm':float(w),'height_mm':float(h),'font_range_pt':[min(fonts),max(fonts)],'vector_svg_has_no_raster':True,'label_bounds_pass':True,
                     'formats':['svg','pdf','eps','png600','RGBtiff600','RGBjpg300'],'caption_file':stem+'.caption.txt','artifact_files':{ext:stem+'.'+ext for ext in ['svg','pdf','eps','png','tiff','jpg']},'plotted_data_file':'paired_difference_source_rows.csv' if metadata['view']=='paired_difference' else 'absolute_core_source_rows.csv'})
    (out/f'{stem}.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with Image.open(out/f'{stem}.png') as im:gray=ImageOps.grayscale(im.convert('RGB'))
    gray.thumbnail((1200,1400));gray.save(out/f'{stem}.grayscale.png')
    plt.close(fig)
    return metadata


def absolute(core,scope,branch,out,*,detail,all_inputs=False):
    outcomes=OUTCOMES[scope];fig,axes=canvas(len(outcomes));inputs=INPUTS if all_inputs else ['preop_only','preop_and_all_waveforms']
    clipping=[]
    for ri,(outcome,olabel) in enumerate(outcomes):
      whole=core[core.outcome.eq(outcome)]
      if detail:
        ymin,ymax=DETAIL_LIMITS[outcome]
        # Detail limits always include all displayed model point estimates.
        observed=whole[whole.feature_set.isin(inputs)].model*100
        ymin=min(ymin,float(observed.min())-.2);ymax=max(ymax,float(observed.max())+.2)
      else:
        vals=np.r_[whole[whole.feature_set.isin(inputs)].model.to_numpy()*100,whole.treat_all.to_numpy()*100,0]
        span=vals.max()-vals.min();ymin=float(vals.min()-.04*span);ymax=float(vals.max()+.06*span)
      for ci,(family,flabel) in enumerate(FAMILIES):
        ax=axes[ri,ci];group=whole[whole.model_type.eq(family)]
        for inp in inputs:
          z=group[group.feature_set.eq(inp)].sort_values('threshold_probability')
          ax.plot(z.threshold_probability*100,z.model*100,color=COLORS[inp],ls=STYLES[inp],lw=ps.CURVE_WIDTH)
        reference=group[group.feature_set.eq('preop_only')].sort_values('threshold_probability')
        x=reference.threshold_probability.to_numpy()*100;all_nb=reference.treat_all.to_numpy()*100
        ax.plot(x,all_nb,color=ps.color('reference'),ls='-.',lw=ps.REFERENCE_WIDTH,zorder=1)
        ax.axhline(0,color=ps.color('ink'),lw=ps.REFERENCE_WIDTH,zorder=1)
        omitted=int((all_nb<ymin).sum())
        if omitted:
          # This is explicit vertical detail, never a silent truncated reference.
          xcross=float(np.interp(ymin,all_nb[::-1],x[::-1]))
          ax.annotate('',xy=(xcross,ymin+.015*(ymax-ymin)),xytext=(xcross,ymin+.14*(ymax-ymin)),
                      arrowprops={'arrowstyle':'-|>','lw':.7,'color':ps.color('reference')})
        clipping.append({'outcome':outcome,'family':family,'model_points_below':int(((group[group.feature_set.isin(inputs)].model*100)<ymin).sum()),
                         'treat_all_points_below_detail_axis':omitted,'explicit_down_arrow':bool(omitted),'y_limits_per100':[ymin,ymax]})
        ax.set(xlim=(x.min(),x.max()),ylim=(ymin,ymax),xticks=XTICKS[outcome])
        ax.yaxis.set_major_locator(MaxNLocator(4,prune='both'));style_axes(ax)
        title(ax,chr(97+ri*3+ci),olabel+' · '+flabel)
        if ci==0:ax.set_ylabel('Net benefit per 100 operations')
        else:ax.tick_params(labelleft=False)
        if ri==len(outcomes)-1:ax.set_xlabel('Threshold probability (%)')
    legend(fig,inputs,detail=detail)
    stem='S5A_dca_absolute_detail' if detail else ('DCA_all_inputs_full_range' if all_inputs else 'S5A_dca_absolute_full_range')
    note=CONVENTIONAL_NOTE if detail else (ALL_INPUTS_NOTE if all_inputs else FULL_NOTE)
    if scope!='main_shared' or branch!='non_windowed':note=f'{SCOPE_NAMES[scope]}; {BRANCH_NAMES[branch]}. '+note
    if scope=='mortality_waveform_sensitivity':note='Exploratory meeting backup; source-label linkage remains unresolved. '+note
    return save(fig,stem,out,note,{'view':'absolute_detail' if detail else 'absolute_full_range','cohort_scope':scope,'branch':branch,'displayed_inputs':inputs,
             'number_model_curves_per_panel':len(inputs),'number_reference_curves_per_panel':2,'intervals_plotted':False,'detail_clipping':clipping})


def difference(delta,scope,branch,out):
    outcomes=OUTCOMES[scope];fig,axes=canvas(len(outcomes))
    for ri,(outcome,olabel) in enumerate(outcomes):
      whole=delta[delta.outcome.eq(outcome)]
      ymin=min(0,float(whole.lower.min()*100));ymax=max(0,float(whole.upper.max()*100));pad=.08*(ymax-ymin)
      for ci,(family,flabel) in enumerate(FAMILIES):
        ax=axes[ri,ci];z=whole[whole.candidate_model.eq(family)].sort_values('threshold_probability');x=z.threshold_probability.to_numpy()*100
        ax.fill_between(x,z.lower.to_numpy()*100,z.upper.to_numpy()*100,color=ps.color('combined_interval'),lw=0)
        ax.plot(x,z.delta.to_numpy()*100,color=COLORS['preop_and_all_waveforms'],lw=ps.CURVE_WIDTH)
        ax.axhline(0,color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH)
        ax.set(xlim=(x.min(),x.max()),ylim=(ymin-pad,ymax+pad),xticks=XTICKS[outcome])
        ax.yaxis.set_major_locator(MaxNLocator(4,prune='both'));style_axes(ax)
        title(ax,chr(97+ri*3+ci),olabel+' · '+flabel)
        if ci==0:ax.set_ylabel('Δ net benefit per 100 operations')
        else:ax.tick_params(labelleft=False)
        if ri==len(outcomes)-1:ax.set_xlabel('Threshold probability (%)')
    legend(fig,[],delta=True)
    note=DELTA_NOTE
    if scope!='main_shared' or branch!='non_windowed':note=f'{SCOPE_NAMES[scope]}; {BRANCH_NAMES[branch]}. '+note
    if scope=='mortality_waveform_sensitivity':note='Exploratory meeting backup; source-label linkage remains unresolved. '+note
    return save(fig,'S5B_dca_paired_difference',out,note,{'view':'paired_difference','cohort_scope':scope,'branch':branch,'comparison':'Combined minus same-family clinical only',
             'number_model_comparison_curves_per_panel':1,'intervals_plotted':True,'interval_type':'Nominal pointwise95% patient-cluster bootstrap conditional on saved fits'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=OUTCOMES,default='main_shared')
    parser.add_argument('--branch',choices=['non_windowed','windowed'],default='non_windowed')
    args=parser.parse_args()
    scope,branch=args.scope,args.branch
    out=HERE/'figures' if (scope,branch)==('main_shared','non_windowed') else HERE/'figures'/scope/branch
    ps.apply_mpl_style()
    core,delta,checks=load(scope,branch)
    out.mkdir(parents=True,exist_ok=True)
    core.to_csv(out/'absolute_core_source_rows.csv',index=False)
    delta.to_csv(out/'paired_difference_source_rows.csv',index=False)
    meta=[absolute(core,scope,branch,out,detail=True),absolute(core,scope,branch,out,detail=False),absolute(core,scope,branch,out,detail=False,all_inputs=True),difference(delta,scope,branch,out)]
    provenance=json.loads((HERE/'source_provenance.json').read_text())
    for source in provenance['files']:assert digest(HERE/source['portable_copy'])==source['sha256']
    (out/'dca_manifest.json').write_text(json.dumps({'scope':scope,'branch':branch,'scientific_policy':'Saved source point estimates and intervals replayed exactly; no new fitting or bootstrap.','figures':meta,
              'validation':checks,'source_hashes_preserved':True,'runtime':{p:version(p) for p in ['numpy','pandas','matplotlib','Pillow']},'python':platform.python_version()},indent=2)+'\n')
    thumbs=[]
    for item in meta:
      with Image.open(out/(item['stem']+'.jpg')) as im:
        im=im.copy();im.thumbnail((640,620))
      tile=Image.new('RGB',(680,670),'white');tile.paste(im,((680-im.width)//2,35));ImageDraw.Draw(tile).text((15,10),item['stem'],fill='black');thumbs.append(tile)
    sheet=Image.new('RGB',(1360,1340),'#eeeeee')
    for i,tile in enumerate(thumbs):sheet.paste(tile,((i%2)*680,(i//2)*670))
    sheet.save(out/'dca_contact_sheet.jpg',quality=ps.JPEG_QUALITY)
    print(json.dumps({'scope':scope,'branch':branch,'absolute_core_source_rows':len(core),'paired_rows':len(delta),'figures':len(meta),'output':str(out)},indent=2))

if __name__=='__main__':main()
