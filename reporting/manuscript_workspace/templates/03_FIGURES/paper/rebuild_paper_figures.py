"""Re-render exact mentor-packet plotted data; no model fitting or canonical writes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '05_CODE'))
import publication_style as ps
from pathlib import Path
import hashlib, json, shutil, textwrap, subprocess, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

OUT = Path(__file__).resolve().parent
MAIN, SUP = OUT/'inputs/main', OUT/'inputs/supplement'
MM = 1 / 25.4
COLORS = [ps.INPUT_COLORS[key] for key in ps.INPUT_ORDER]
FOLD_COLORS = ps.FOLD_COLORS
STYLES = [ps.INPUT_STYLES[key] for key in ps.INPUT_ORDER]
FOLD_STYLES = ps.FOLD_STYLES
FAMILIES = [('logistic_regression', 'LR'), ('xgboost', 'XGBoost'), ('ebm', 'EBM')]
OUTCOMES = [('any_aki', 'AKI'), ('icu_admission', 'ICU admission')]
INPUTS = list(ps.INPUT_ORDER)
INPUT_LABELS = [ps.INPUT_LABELS[key] for key in ps.INPUT_ORDER]
ps.apply_mpl_style()
manifest = []
READER_TITLES = {'figure_1_cohort_workflow': 'Cohort selection', 'figure_2_precision_recall': 'Precision–recall curves', 'figure_3_incremental_ap': 'Incremental average precision', 'figure_4_ebm_response_functions': 'EBM response functions', 'S2-windowed-incremental-ap': 'Windowed AP differences', 'S3-broader-icu-incremental-ap': 'Broader-cohort ICU AP differences', 'S4-whole-case-calibration': 'Calibration, all inputs', 'S6-ebm-nonlinearity-missingness': 'EBM functions and missing values'}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def clean(ax, grid=True):
    ax.spines[['top','right']].set_visible(False)
    if grid: ax.grid(axis='y', color=ps.color('grid'), lw=ps.GRID_WIDTH)
    ax.set_axisbelow(True)
def title(ax, letter, label): ax.set_title(r'$\bf{'+letter+'}$  '+label, loc='left', pad=5)
def canvas(height): return plt.figure(figsize=(ps.WIDTH_MM*MM,height*MM))
def input_legend(fig, baseline=False):
    handles=[Line2D([0],[0],color=c,ls=s,lw=ps.CURVE_WIDTH,label=l) for c,s,l in zip(COLORS,STYLES,INPUT_LABELS)]
    if baseline: handles.append(Line2D([0],[0],color=ps.color('reference'),ls=ps.REFERENCE_STYLE,lw=ps.REFERENCE_WIDTH,label='Prevalence'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=4 if baseline else 3,frameon=False,handlelength=2.6,columnspacing=1.6)
def folds_legend(fig):
    fig.legend(handles=[Line2D([0],[0],color=c,ls=s,lw=ps.CURVE_WIDTH,label=f'Fold {i+1}') for i,(c,s) in enumerate(zip(FOLD_COLORS,FOLD_STYLES))],loc='lower center',bbox_to_anchor=(.5,.005),ncol=5,frameon=False,handlelength=2.1,columnspacing=1)

def save(fig, stem, sources, caption, extras=()):
    # Deliberately no tight bounding-box crop: exported physical dimensions are exact.
    fig.canvas.draw()
    width,height=fig.get_size_inches()*25.4
    assert abs(width-ps.WIDTH_MM)<1e-8 and height<=234
    texts=[t for t in fig.findobj(matplotlib.text.Text) if t.get_text() and t.get_visible()]
    fonts=sorted(set(float(t.get_fontsize()) for t in texts))
    assert min(fonts)>=8 and max(fonts)<=12, (stem,fonts)
    visible_text=list(fig.texts)
    all_axes=[]
    def descend(axis):
        all_axes.append(axis)
        for child in axis.child_axes:descend(child)
    for axis in fig.axes:descend(axis)
    for axis in all_axes:
        visible_text.extend(axis.texts)
        visible_text.extend([axis.title,axis._left_title,axis._right_title])
        if axis.axison:
            visible_text.extend([axis.xaxis.label,axis.yaxis.label])
            for positions,labels,limits in [(axis.get_xticks(),axis.get_xticklabels(),axis.get_xlim()),
                                            (axis.get_yticks(),axis.get_yticklabels(),axis.get_ylim())]:
                visible_text.extend(t for pos,t in zip(positions,labels) if min(limits)-1e-10<=pos<=max(limits)+1e-10)
    for legend in fig.legends:visible_text.extend(legend.get_texts())
    renderer=fig.canvas.get_renderer()
    visible_text=[t for t in visible_text if t.get_text() and t.get_visible()]
    outside=[]
    for text in visible_text:
        box=text.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:
            outside.append(text.get_text())
    assert not outside,(stem,'off-canvas labels',outside)
    for ext,dpi in [('svg',None),('pdf',None),('eps',None),('png',ps.PNG_DPI),('tiff',ps.TIFF_DPI),('jpg',ps.JPG_DPI)]:
        kw={'dpi':dpi} if dpi else {}
        if ext=='jpg': kw['pil_kwargs']={'quality':ps.JPEG_QUALITY,'subsampling':0}
        if ext=='tiff': kw['pil_kwargs']={'compression':ps.TIFF_COMPRESSION}
        fig.savefig(OUT/f'{stem}.{ext}',**kw)
        if ext == 'tiff':
            with Image.open(OUT/f'{stem}.{ext}') as raster:
                rgb = raster.convert('RGB')
            rgb.save(OUT/f'{stem}.{ext}', compression=ps.TIFF_COMPRESSION, dpi=(ps.TIFF_DPI,ps.TIFF_DPI))
    plt.close(fig)
    copied=[]
    for source in sources:
        source=Path(source)
        dest=OUT / (stem+'.plotted.csv' if len(sources)==1 and source.suffix=='.csv' else source.name)
        if source.resolve()!=dest.resolve(): shutil.copyfile(source,dest)
        copied.append({'source':str(source),'sha256':sha(source),'local_copy':dest.name})
    for p in extras:
        p=Path(p); shutil.copyfile(p,OUT/p.name)
        copied.append({'source':str(p),'sha256':sha(p),'local_copy':p.name})
    caption_text=Path(caption).read_text().strip()
    # Reader captions carry meaning; operational paths remain in source metadata.
    caption_text=re.split(r'\n\s*Source artifacts? for audit:',caption_text,maxsplit=1)[0].strip()
    caption_text=re.sub(r'^(?:Supplementary\s+)?Figure\s+S?\d+[.:]\s*','',caption_text)
    for letter in 'ABCD': caption_text=caption_text.replace(f'({letter})',f'({letter.lower()})')
    caption_path=OUT/f'{stem}.caption.txt'
    if not caption_path.exists() or caption_path.read_text().strip()!=caption_text.strip():
        caption_path.write_text(caption_text+'\n')
    # Check SVG does not embed bitmap marks.
    svg=(OUT/f'{stem}.svg').read_text(); assert '<image' not in svg
    entry={'style_profile_sha256':ps.STYLE_SHA256,'stem':stem,'title':READER_TITLES[stem],'width_mm':float(width),'height_mm':float(height),'font_family':ps.FONT_FAMILY,
      'font_sizes_pt':fonts,'minimum_rule_width_pt':.4,'svg_has_embedded_raster':False,'drawn_text_bounds_verified':True,'drawn_text_count':len(visible_text),
      'formats':{'svg':'editable vector scientific master','pdf':'vector master with embedded fonts',
       'eps':'publisher-preferred vector copy; transparency flattened to equivalent opaque tints',
       'tiff':'publisher raster combination copy, 600 dpi, lossless LZW compression',
       'png':'600 dpi lossless publication/review raster','jpg':'300 dpi quality-95 convenience preview'},
      'sources':copied,'caption_file':f'{stem}.caption.txt'}
    manifest.append(entry)
    (OUT/f'{stem}.metadata.json').write_text(json.dumps(entry,indent=2)+'\n')
    print(stem,round(height,1),'mm',flush=True)

# Main Figure 1: restore the historical retained spine and explicit exclusions.
import sys
from dataclasses import replace
from cohort_support import CohortStage, _draw_cohort_flow_fallback
flow=json.loads((MAIN/'figure_1_counts.json').read_text())
labels=['VitalDB operations','General anesthesia','ASA eligibility',
        'All four waveforms available','Shared AKI/ICU eligibility','Final analysis cohort']
stages=[replace(CohortStage(**stage),title=label,detail=None,
                removal_reason=stage.get('removal_reason', '').replace('No retained waveform-feature record','No waveform-feature record') if stage.get('removal_reason') else None)
        for stage,label in zip(flow['stages'],labels)]
fig=_draw_cohort_flow_fallback(stages,None,None,publication=True)
# Keep outcome annotations together: AKI and ICU are not mutually exclusive.
w,h=fig.get_size_inches(); footer=.66
fig.set_size_inches(w,h+footer)
fig.axes[0].set_position([0,footer/(h+footer),1,h/(h+footer)])
fig.text(.5,.41/(h+footer),'Outcomes (may overlap)',
         ha='center',va='center',fontsize=8.7,color=ps.color('ink'))
fig.text(.5,.21/(h+footer),'AKI: 189 operations (5.3%)     ICU admission: 834 operations (23.4%)',
         ha='center',va='center',fontsize=ps.STYLE['figure']['title_pt'],color=ps.color('ink'))
caption=OUT/'figure_1_cohort_workflow.caption.txt'
caption.write_text('Cohort selection. Boxes show operations and unique patients; side branches show exclusions. ASA V/VI were excluded; missing ASA was retained. ASA, American Society of Anesthesiologists physical status.\n')
flow_csv=OUT/'figure_1_cohort_workflow.plotted.csv'
pd.DataFrame(flow['stages']).to_csv(flow_csv,index=False)
save(fig,'figure_1_cohort_workflow',[flow_csv,MAIN/'figure_1_counts.json'],caption)

# Main Figure 2: all18 saved PR curves; common outcome-row/family-column grid.
d=pd.read_csv(MAIN/'figure_2_precision_recall.csv'); m=pd.read_csv(MAIN/'figure_2_metrics.csv')
assert len(m)==18
fig=canvas(151)
grid=fig.add_gridspec(2,3,left=.10,right=.98,bottom=.16,top=.98,hspace=.29,wspace=.31)
for ri,(outcome,olabel) in enumerate(OUTCOMES):
 for ci,(family,flabel) in enumerate(FAMILIES):
    cell=grid[ri,ci].subgridspec(2,1,height_ratios=[.25,1],hspace=.08)
    head=fig.add_subplot(cell[0]); head.set(xlim=(0,1),ylim=(0,1));head.axis('off')
    head.text(0,.92,r'$\bf{'+chr(97+ri*3+ci)+'}$  '+f'{olabel} · {flabel}',fontsize=ps.STYLE['figure']['title_pt'],va='top')
    head.text(0,.12,'AP',fontsize=ps.STYLE['figure']['body_pt'],va='center')
    ax=fig.add_subplot(cell[1])
    vals=m[(m.family==family)&(m.outcome==outcome)].set_index('feature_set')
    for j,(inp,color,style) in enumerate(zip(INPUTS,COLORS,STYLES)):
        z=d[(d.family==family)&(d.outcome==outcome)&(d.feature_set==inp)].sort_values('point')
        assert not z.empty
        ax.step(z.recall,z.precision,where='post',color=color,ls=style,lw=ps.CURVE_WIDTH)
        x=.10+j*.30
        head.plot([x,x+.07],[.12,.12],color=color,ls=style,lw=ps.CURVE_WIDTH)
        head.text(x+.09,.12,f'{vals.loc[inp,"average_precision"]:.3f}',fontsize=ps.STYLE['figure']['body_pt'],va='center')
    ax.axhline(vals.prevalence.iloc[0],color=ps.color('reference'),ls=ps.REFERENCE_STYLE,lw=ps.REFERENCE_WIDTH)
    ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1]); clean(ax)
    if ri==1: ax.set_xlabel('Recall (sensitivity)')
    else:ax.tick_params(axis='x',labelbottom=False)
    if ci==0: ax.set_ylabel('Precision (PPV)')
    else:ax.tick_params(axis='y',labelleft=False)
input_legend(fig,True)
save(fig,'figure_2_precision_recall',[MAIN/'figure_2_precision_recall.csv'],MAIN/'figure_2_caption.txt',[MAIN/'figure_2_metrics.csv',MAIN/'figures_2_4_sources.json',MAIN/'figures_2_4_verification.csv'])

def paired_two_panel(data,stem,labels,caption,source,height=96,groups='outcome'):
    fig=canvas(height)
    grid=fig.add_gridspec(1,2,left=.13,right=.98,bottom=.23,top=.85,wspace=.24)
    low=min(0,float(data.lower.min()*100)); high=max(0,float(data.upper.max()*100)); pad=max(1,(high-low)*.1)
    for col,(group,label) in enumerate(labels):
        cell=grid[col].subgridspec(1,2,width_ratios=[1,.99],wspace=.08)
        ax=fig.add_subplot(cell[0]); nums=fig.add_subplot(cell[1],sharey=ax)
        title(ax,chr(97+col),label)
        nums.set_axis_off(); nums.text(1,1.035,'Δ AP (95% CI)',transform=nums.transAxes,ha='right',va='bottom',fontsize=ps.STYLE['figure']['body_pt'])
        for row,(family,fname) in enumerate(FAMILIES):
            z=data[(data[groups]==group)&(data.candidate_model==family)]
            assert len(z)==1
            r=z.iloc[0]; v,l,u=100*np.asarray([r.delta,r.lower,r.upper]); y=2-row
            ax.errorbar(v,y,xerr=[[v-l],[u-v]],fmt='o',color=COLORS[2],capsize=2.3,lw=1,ms=4.4)
            label=f'{v:+.1f} ({l:+.1f}, {u:+.1f})'.replace('-','−')
            nums.text(1,y,label,transform=nums.get_yaxis_transform(),ha='right',va='center',fontsize=ps.STYLE['figure']['body_pt'])
        ax.axvline(0,color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH)
        ax.set(xlim=(low-pad,high+pad),ylim=(-.45,2.55),yticks=[2,1,0],yticklabels=[x[1] for x in FAMILIES])
        ax.tick_params(axis='y',length=0,labelleft=col==0); ax.xaxis.set_major_locator(MaxNLocator(3)); clean(ax,False)
        ax.grid(axis='x',color=ps.color('grid'),lw=ps.GRID_WIDTH); ax.spines['left'].set_visible(False)
    fig.supxlabel('Combined − clinical AP (percentage points)',fontsize=ps.STYLE['figure']['body_pt'],y=.06)
    save(fig,stem,[source],caption)

paired_two_panel(pd.read_csv(MAIN/'figure_3_incremental_ap.csv'),'figure_3_incremental_ap',OUTCOMES,MAIN/'figure_3_caption.txt',MAIN/'figure_3_incremental_ap.csv')

def response_panel(fig,slot,values,hist,letter,label,xlabel,ycol,ref,foldbase=0):
    sg=slot.subgridspec(2,1,height_ratios=[3.7,1],hspace=.035)
    ax=fig.add_subplot(sg[0]); hx=fig.add_subplot(sg[1],sharex=ax)
    title(ax,letter,label)
    for fold,z in values.groupby('fold'):
        i=int(fold)-foldbase; z=z.sort_values('value')
        ax.step(z.value,z[ycol],where='post',color=FOLD_COLORS[i],ls=FOLD_STYLES[i],lw=ps.CURVE_WIDTH)
    ax.axhline(0,color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH); ax.axvline(ref,color=ps.color('reference'),ls=':',lw=ps.REFERENCE_WIDTH)
    ax.set_ylabel('Δ raw log-odds'); ax.tick_params(axis='x',labelbottom=False); clean(ax)
    hx.bar(hist.bin_lower,hist['count'],width=hist.bin_upper-hist.bin_lower,align='edge',color=ps.color('support'),linewidth=0)
    ref_text=f'{ref:.3g}'.replace('-','−')
    hx.set(xlim=(hist.bin_lower.min(),hist.bin_upper.max()),ylabel='Training n',xlabel=xlabel+'\nReference = '+ref_text)
    hx.yaxis.set_major_locator(MaxNLocator(2,integer=True)); hx.xaxis.set_major_locator(MaxNLocator(4)); clean(hx,False)
    return ax,hx

v=pd.read_csv(MAIN/'figure_4_ebm_response_functions.csv'); h=pd.read_csv(MAIN/'figure_4_training_support.csv')
fig=canvas(185); grid=fig.add_gridspec(2,2,left=.11,right=.98,bottom=.15,top=.96,hspace=.50,wspace=.34)
spec=[('A','AKI · Hemoglobin','Hemoglobin (g/dL)'),('B','AKI · Airway pressure','Rescaled-range fluctuation'),('C','ICU admission · Age','Age (years)'),('D','ICU admission · CO₂','Time-reversibility statistic')]
for i,(panel,label,xlabel) in enumerate(spec):
    z=v[v.panel==panel]; hist=h[h.panel==panel]
    response_panel(fig,grid[i//2,i%2],z,hist,chr(97+i),label,xlabel,'raw_contribution_minus_reference',z.reference.iloc[0])
folds_legend(fig)
save(fig,'figure_4_ebm_response_functions',[MAIN/'figure_4_ebm_response_functions.csv',MAIN/'figure_4_training_support.csv'],MAIN/'figure_4_caption.txt',[MAIN/'figure_4_missing_bins.csv',MAIN/'figure_4_verification.csv',MAIN/'figure_4_aki_source_curves.json',MAIN/'figure_4_icu_source_curves.json'])
shutil.copyfile(MAIN/'figure_4_ebm_response_functions.csv',OUT/'figure_4_ebm_response_functions.plotted.csv')

for stem,labels,groups in [('S2-windowed-incremental-ap',OUTCOMES,'outcome'),('S3-broader-icu-incremental-ap',[('non_windowed','Whole case'),('windowed','Windowed')],'branch')]:
    paired_two_panel(pd.read_csv(SUP/f'{stem}.plotted.csv'),stem,labels,SUP/f'{stem}.caption.txt',SUP/f'{stem}.plotted.csv',groups=groups)
    shutil.copyfile(SUP/f'{stem}.sources.json',OUT/f'{stem}.sources.json')

# S4: honest full0–1 probability scales plus explicitly marked AKI detail insets.
stem='S4-whole-case-calibration'; d=pd.read_csv(SUP/f'{stem}.plotted.csv')
assert len(d)==180
fig=canvas(151); axes=fig.subplots(2,3,sharex=True,sharey=True)
fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.94,hspace=.37,wspace=.31)
for ri,(outcome,olabel) in enumerate(OUTCOMES):
 for ci,(family,flabel) in enumerate(FAMILIES):
    ax=axes[ri,ci]; title(ax,chr(97+ri*3+ci),f'{olabel} · {flabel}')
    for inp,color,style,marker in zip(INPUTS,COLORS,STYLES,[ps.INPUT_MARKERS[key] for key in INPUTS]):
        z=d[(d.outcome==outcome)&(d.family==family)&(d.feature_set==inp)].sort_values('bin')
        assert len(z)==10
        ax.plot(z.mean_risk,z.observed_rate,color=color,ls=style,marker=marker,ms=3,lw=ps.CURVE_WIDTH)
    ax.plot([0,1],[0,1],color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH)
    ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1]); ax.set_aspect('equal'); clean(ax)
    if ri==0:
        detail=ax.inset_axes([.49,.43,.48,.43])
        for inp,color,style,marker in zip(INPUTS,COLORS,STYLES,[ps.INPUT_MARKERS[key] for key in INPUTS]):
            z=d[(d.outcome==outcome)&(d.family==family)&(d.feature_set==inp)].sort_values('bin')
            detail.plot(z.mean_risk,z.observed_rate,color=color,ls=style,marker=marker,ms=2,lw=0.75 * ps.CURVE_WIDTH)
        detail.plot([0,.35],[0,.35],color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH)
        detail.set(xlim=(0,.35),ylim=(0,.35),xticks=[0,.3],yticks=[0,.3])
        detail.tick_params(labelsize=8,length=1.5,pad=1)
        detail.set_title('0–0.35',fontsize=ps.STYLE['figure']['body_pt'],pad=3)
        detail.spines[['top','right']].set_visible(False)
        for spine in detail.spines.values():spine.set_linewidth(.5)
        ax.add_patch(Rectangle((0,0),.35,.35,fill=False,edgecolor=ps.color('reference'),lw=.5,ls=':'))
    if ri==1: ax.set_xlabel('Mean predicted risk')
    if ci==0: ax.set_ylabel('Observed risk')
input_legend(fig)
save(fig,stem,[SUP/f'{stem}.plotted.csv'],SUP/f'{stem}.caption.txt',[SUP/f'{stem}.sources.json'])

stem='S6-ebm-nonlinearity-missingness'; d=pd.read_csv(SUP/f'{stem}.plotted.csv')
fig=canvas(185); grid=fig.add_gridspec(2,2,left=.115,right=.98,bottom=.14,top=.96,hspace=.57,wspace=.35,height_ratios=[1.5,1])
for i,(name,label,xlabel) in enumerate([('AKI · Observed WBC response','AKI · WBC','WBC (10³/µL)'),('ICU · Processed pleth mean response','ICU admission · PLETH mean','Filtered waveform mean')]):
    z=d[d.panel==name]; hist=d[d.panel==name+' training histogram']
    response_panel(fig,grid[0,i],z,hist,chr(97+i),label,xlabel,'raw_contribution',z.reference_value.iloc[0],foldbase=1)
for i,(name,label) in enumerate([('AKI · Missing WBC contribution','AKI · Missing WBC'),('ICU · Missing CRP contribution','ICU admission · Missing CRP')]):
    ax=fig.add_subplot(grid[1,i]); z=d[d.panel==name].sort_values('fold'); title(ax,chr(99+i),label)
    ax.axhline(0,color=ps.color('reference'),ls='--',lw=ps.REFERENCE_WIDTH)
    for j,r in enumerate(z.itertuples()):
        ax.plot(r.fold,r.raw_contribution,marker=ps.FOLD_MARKERS[j],ms=5,color=FOLD_COLORS[j],ls='none')
    ax.set(xlabel='Outer fold (missing training n)',ylabel='Missing-bin raw log-odds',xticks=[1,2,3,4,5],xlim=(.55,5.45))
    ax.set_xticklabels([f'{int(r.fold)}\n({int(r.missing_training_n):,})' for r in z.itertuples()])
    ax.margins(y=.35); clean(ax)
folds_legend(fig)
save(fig,stem,[SUP/f'{stem}.plotted.csv'],SUP/f'{stem}.caption.txt',[SUP/f'{stem}.sources.json',SUP/'S6-aki-curves.json',SUP/'S6-icu-curves.json'])

# Review companions: PDF at native size; grayscale proof; compact contact sheet.
graydir=OUT/'qa'; graydir.mkdir(exist_ok=True)
thumbs=[]
for item in manifest:
    im=Image.open(OUT/f'{item["stem"]}.png').convert('RGB')
    im=ImageOps.grayscale(im)
    # 150 dpi grayscale proof retains physical dimensions in metadata.
    native=(round(item['width_mm']/25.4*150),round(item['height_mm']/25.4*150))
    im=im.resize(native,Image.Resampling.LANCZOS)
    im.save(graydir/f'{item["stem"]}.grayscale.png',dpi=(150,150))
    thumb=im.copy(); thumb.thumbnail((510,700))
    tile=Image.new('RGB',(550,750),'white'); tile.paste(thumb,((550-thumb.width)//2,35))
    ImageDraw.Draw(tile).text((15,12),item['stem'],fill='black'); thumbs.append(tile)
sheet=Image.new('RGB',(550*5,750*2),ps.color('grid'))
for i,tile in enumerate(thumbs): sheet.paste(tile,((i%5)*550,(i//5)*750))
sheet.save(graydir/'grayscale_contact_sheet.jpg',quality=ps.JPEG_QUALITY)
subprocess.run(['pdfunite',*[str(OUT/f'{item["stem"]}.pdf') for item in manifest],str(graydir/'native_size_figures.pdf')],check=True)
(OUT/'figure_manifest.json').write_text(json.dumps({'scientific_data_policy':'Exact numerical replay of existing audited plotted data; no fitting, selection, new uncertainty, or canonical artifact writes.','figures':manifest},indent=2)+'\n')
(OUT/'README.txt').write_text('Baseline paper figures; S1 and S5 are rebuilt in their own companion folders.\nAll canvases are exactly 174 mm wide; heights and fonts are in figure_manifest.json. These dimensions are our consistent design convention, not a claimed journal limit.\nSVG, PDF and EPS are true vector masters. PNG and lossless-LZW TIFF are 600 dpi; JPG is a 300 dpi convenience preview.\nNo embedded overall titles or captions; captions, plotted data and source hashes are separate.\nS1 additionally draws its existing saved nominal 95% intervals; no new uncertainty was computed.\nInspect qa/grayscale_contact_sheet.jpg and the native-size PDF proof before submission.\nRebuild: activate aki_prediction_project, then python rebuild_paper_figures.py.\n')
print('Complete:',len(manifest),'publication figures')
