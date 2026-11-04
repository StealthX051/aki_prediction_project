"""Rebuild editable manuscript books using only files inside this workspace."""
from pathlib import Path
from copy import deepcopy
import json,re,subprocess,shutil,textwrap
import pandas as pd
from docx import Document
from publication_style import apply_document_style, CAPTION_PT, WIDTH_MM, STYLE_SHA256
from publication_navigation import bookmark, hyperlink, word_book_footer
from docx.text.paragraph import Paragraph
from docx.shared import Mm,Pt
from docx.enum.section import WD_SECTION_START,WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'01_MANUSCRIPT';OUT.mkdir(exist_ok=True)
AUTHOR=OUT/'author_material';AUTHOR.mkdir(exist_ok=True)

def init(title):
 d=Document();s=d.sections[0];s.page_width=Mm(210);s.page_height=Mm(297);s.top_margin=s.bottom_margin=Mm(13);s.left_margin=s.right_margin=Mm(18)
 apply_document_style(d)
 d.styles['Normal'].paragraph_format.space_after=Pt(4)
 d.core_properties.title=title
 return d

def clean(path):
 s=path.read_text().strip();s=re.split(r'\b(?:Source files|Source artifacts|Source data|Sources|Source):',s,maxsplit=1)[0]
 return re.sub(r'^Figure\s+S?\d+[.:]?\s*','',s).strip()

def entries():
 p=ROOT/'03_FIGURES/paper'
 base={f['stem']:dict(f,folder=str(p.relative_to(ROOT))) for f in json.loads((p/'figure_manifest.json').read_text())['figures']}
 revised=json.loads((ROOT/'03_FIGURES/narrative_revision/figure_manifest.json').read_text())['figures']
 main=[dict(base['figure_1_cohort_workflow'],label='Figure 1',caption_suffix='Predictions use information available by operation end.')]+[f for f in revised if f['label'].startswith('Figure ') and not f['label'].startswith('Figure S')]
 main=sorted(main,key=lambda f:int(f['label'].split()[-1]))
 s1p=ROOT/'03_FIGURES/single_channel_comparison'
 channels=[dict(f,folder=str(s1p.relative_to(ROOT))) for f in json.loads((s1p/'figure_manifest.json').read_text())['figures']]
 roc_p=ROOT/'03_FIGURES/roc_curves'
 roc=[dict(f,folder=str(roc_p.relative_to(ROOT))) for f in json.loads((roc_p/'figure_manifest.json').read_text())['figures']]
 dp=ROOT/'03_FIGURES/decision_curve/figures'
 dca=json.loads((dp/'dca_manifest.json').read_text())['figures']
 primary=[];extra=[]
 for f in dca:
  x=dict(f,folder=str(dp.relative_to(ROOT)),label='Reference '+str(len(extra)+1))
  if f['stem']=='S5A_dca_absolute_detail':primary.append(x)
  elif f['stem']=='S5B_dca_paired_difference':primary.append(x)
  else:extra.append(x)
 supp=[base['figure_2_precision_recall']]+roc+channels+[base['S2-windowed-incremental-ap'],base['S3-broader-icu-incremental-ap']]+primary+[f for f in revised if f['label']=='Figure S9']
 cases=ROOT/'03_FIGURES/heldout_cases/figure_manifest.json'
 if cases.exists():
  for f in json.loads(cases.read_text())['figures']:supp.append(dict(f,folder='03_FIGURES/heldout_cases'))
 for i,f in enumerate(supp,1):f['label']=f'Figure S{i}'
 return main,supp,extra


def main_reading_order():
 main,_,_=entries()
 return [('figure',main[0]),('table',ROOT/'02_TABLES/main/Table_1_cohort.docx'),
         ('table',ROOT/'02_TABLES/main/Table_2_core_comparisons.docx')]+[('figure',f) for f in main[1:]]

def caption_text(f):
 text=clean(ROOT/f['folder']/f['caption_file'])
 return text+(' '+f['caption_suffix'] if f.get('caption_suffix') else '')

def caption(d,f):
 p=d.add_paragraph();p.paragraph_format.space_after=Pt(3)
 r=p.add_run(f['label'].replace('Figure','Fig.')+' ');r.bold=True
 p.add_run(caption_text(f).rstrip('.'))
 for r in p.runs:r.font.size=Pt(CAPTION_PT)
 return p

def figure(d,f,anchor=None):
 p=d.add_paragraph();p.paragraph_format.space_after=Pt(5);p.paragraph_format.keep_with_next=True
 if anchor:bookmark(p,anchor,1000+len(d.paragraphs))
 p.add_run().add_picture(str(ROOT/f['folder']/(f['stem']+'.png')),width=Mm(WIDTH_MM))
 caption(d,f)

def append_doc(d,path):
 src=Document(path)
 # Resolve source font inheritance before merging into a differently styled book.
 for element in src.element.body.iter(qn('w:p')):
  paragraph=Paragraph(element,src);style=paragraph.style
  for run in paragraph.runs:
   for attribute in ('name','size'):
    if getattr(run.font,attribute) is not None:continue
    current=style
    while current is not None:
     value=getattr(current.font,attribute)
     if value is not None:setattr(run.font,attribute,value);break
     current=current.base_style
 for element in src.element.body:
  if element.tag!=qn('w:sectPr'):d.element.body.insert(len(d.element.body)-1,deepcopy(element))

def new_section(d,landscape=False,source=None):
 s=d.add_section(WD_SECTION_START.NEW_PAGE)
 s.page_width=Mm(297 if landscape else 210);s.page_height=Mm(210 if landscape else 297)
 s.orientation=WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
 s.left_margin=s.right_margin=Mm(15 if landscape else 18);s.top_margin=s.bottom_margin=Mm(13)
 if source:
  original=Document(source).sections[0]
  for field in ('page_width','page_height','orientation','left_margin','right_margin','top_margin','bottom_margin','header_distance','footer_distance'):
   setattr(s,field,getattr(original,field))

def books():
 main,supp,extra=entries()
 d=init('Main figures and tables')
 for i,(kind,item) in enumerate(main_reading_order()):
  if kind=='table':
   new_section(d,source=item);first=len(d.paragraphs);append_doc(d,item)
   number=1 if 'Table_1_' in item.name else 2
   bookmark(d.paragraphs[first],f'MainTable_{number}',2000+number)
  else:
   if i:new_section(d)
   figure(d,item,anchor='MainFigure_'+item['label'].split()[-1])
 word_book_footer(d,[('Figures',[(str(i),'MainFigure_'+str(i)) for i in range(1,6)]),('Tables',[(str(i),'MainTable_'+str(i)) for i in range(1,3)])])
 d.save(OUT/'Main_Figures_and_Tables.docx')
 d=init('Supplementary figures and methods')
 append_doc(d,OUT/'Supplementary_Methods.docx')
 bookmark(d.paragraphs[0],'SupplementaryMethods',3000)
 for f in supp:
  d.add_page_break();figure(d,f,anchor='SupplementaryFigure_'+f['label'].split()[-1])
 word_book_footer(d,[('',[('Methods','SupplementaryMethods')]),('Figures',[(f['label'].split()[-1],'SupplementaryFigure_'+f['label'].split()[-1]) for f in supp])])
 d.save(OUT/'Supplementary_Figures_and_Methods.docx')
 d=init('Decision curves: reference views')
 for i,f in enumerate(extra):
  if i:d.add_page_break()
  d.add_heading(f['title'],1)
  figure(d,f)
 d.save(AUTHOR/'DCA_Full_Scale_Reference_Views.docx')
 d=init('Supplementary tables');d.add_heading('Supplementary tables',0)
 d.add_paragraph('Editable tables and numeric sources.')
 for item in json.loads((OUT/'publication_tables/publication_table_index.json').read_text())['tables']:
  d.add_heading(item['title'],2)
  p=d.add_paragraph()
  data_suffix='raw.csv' if (OUT/'publication_tables'/(item['stem']+'_raw.csv')).exists() else 'csv'
  for n,(label,suffix) in enumerate([('Word','docx'),('PDF','pdf'),('Data',data_suffix)]):
   if n:p.add_run(' · ')
   filename=item['stem']+('_raw.csv' if suffix=='raw.csv' else '.'+suffix)
   assert (OUT/'publication_tables'/filename).exists(),filename
   hyperlink(p,label,target='publication_tables/'+filename)
 d.save(OUT/'Supplementary_Tables_Index.docx')
 captions=init('Manuscript figure captions');captions.add_heading('Figure captions',0)
 for f in main+supp:
  caption(captions,f)
 captions.save(OUT/'Figure_Captions.docx')
 (ROOT/'03_FIGURES/active_figure_index.json').write_text(json.dumps({'main':main,'supplementary':supp,'additional_reference':extra},indent=2)+'\n')


def methods():
 src=ROOT/'01_MANUSCRIPT/frozen_definitions/METHODS.md'
 d=init('Frozen scientific definitions');d.add_heading('Frozen scientific definitions',0)
 d.add_paragraph('Repository METHODS snapshot. Operational details are retained for provenance; the manuscript starting document below is an abbreviated writing aid.')
 code=False
 for line in src.read_text().splitlines():
  if line.startswith('```'):code=not code;continue
  if line.startswith('#'):d.add_heading(line.lstrip('# ').strip(),min(3,len(line)-len(line.lstrip('#'))))
  elif line.strip():
   p=d.add_paragraph(line)
   if code:
    for r in p.runs:r.font.name='Consolas';r.font.size=Pt(8)
 d.save(AUTHOR/'Frozen_Methods_Reference.docx')
 d=init('Supplementary methods');d.add_heading('Supplementary methods',0)
 for i,(heading,text) in enumerate(method_blocks()):
  if i==3:d.add_page_break()
  d.add_heading(heading,1);d.add_paragraph(text)
 d.save(OUT/'Supplementary_Methods.docx')
 (OUT/'Supplementary_Methods.txt').write_text('Supplementary methods\n\n'+'\n\n'.join(h+'\n'+t for h,t in method_blocks())+'\n')

def method_blocks():
 return [
 ('Population and outcomes','The main shared cohort comprised 3,571 operations in 3,432 patients undergoing general anesthesia with ECG, plethysmography (PLETH), carbon dioxide (CO₂) and airway pressure (AWP) recordings. ASA physical status V/VI was excluded; missing ASA and repeat operations were retained. Acute kidney injury (AKI) required a creatinine rise ≥0.3 mg/dL within 48 hours or ≥1.5 times baseline within 72 hours after operation end. Baseline was the source preoperative creatinine, restricted to finite values >0 and ≤4 mg/dL. Urine-output and renal-replacement criteria were unavailable. Intensive care unit (ICU) admission was positive source ICU duration, without distinction between planned and unplanned admission. The broader ICU sensitivity cohort comprised 5,765 operations in 5,543 patients.'),
 ('Waveform and clinical inputs','Catch24 descriptors were extracted after operative cropping, signal cleaning, filtering and resampling. Whole-case inputs used complete operative traces at 10 Hz, yielding 96 descriptors across four channels. Windowed inputs used complete 10-second windows with a nominal five-second stride and four aggregates per descriptor, yielding 384 features. Windowed ECG and PLETH retained 100 Hz; CO₂ and AWP retained 62.5 Hz. Branch comparisons therefore changed both sampling rate and temporal aggregation. Clinical inputs comprised preoperative demographics, comorbidity, surgical information and laboratory values. Continuous preoperative predictors were capped at the training partition’s 0.5th and 99.5th percentiles.'),
 ('Model development and calibration','L2 logistic regression (LR), XGBoost and an additive explainable boosting machine (EBM; interactions disabled) were evaluated using five patient-grouped outer folds. All operations from a patient remained together in every split. Hyperparameter optimization used three inner folds and raw-probability average precision, with 20 trials for LR and 50 for each tree model. LR used training medians, training-derived missingness indicators and scaling; XGBoost and EBM handled missing values natively. Following hyperparameter selection, five calibration folds generated out-of-fold predictions within the outer training partition. A regularized logistic recalibrator was fitted to their logits, and Youden’s J selected the classification threshold from their calibrated values. The base model was then refitted on all outer training data; the fitted recalibrator and threshold were applied to the held-out outer fold. Hyperparameter selection preceded calibration cross-fitting; calibration-training predictions were development data. Outer test patients were excluded from learned preprocessing, hyperparameter selection, calibration fitting and threshold selection.'),
 ('Evaluation','AKI was the primary endpoint; ICU admission was secondary. The six whole-case combined-versus-same-family clinical average precision (AP) contrasts in the main shared cohort were the primary model comparisons. Other core contrasts were secondary; windowed, broader-cohort and ablation analyses were exploratory. AP, area under the receiver operating characteristic curve (AUROC), Brier score, sensitivity, specificity, precision, F1 score and accuracy were evaluated using pooled outer-test predictions and saved fold-specific classification thresholds. Nominal 95% percentile confidence intervals used 5,000 shared unstratified patient-cluster bootstrap draws (seed 0), preserving all operations for sampled patients. Paired contrasts used the same draws. These intervals condition on the saved fitted models; the full model-development procedure was not repeated. No multiplicity adjustment was applied.'),
 ('Calibration and decision curves','Calibration plots used pooled held-out probabilities in up to ten equal-frequency bins, collapsing tied bin edges. Exploratory decision curves used threshold probabilities from 1% to 30% for AKI and 5% to 60% for ICU admission, in one-percentage-point increments. Net benefit was TP/N − (FP/N) × pₜ/(1 − pₜ), reported per 100 operations (TP, true positives; FP, false positives; N, operations; pₜ, threshold probability). Conventional curves compare clinical and combined models with treat-all and treat-none policies; paired curves report combined-minus-clinical net benefit with pointwise 95% patient-cluster bootstrap confidence intervals. Table S5 uses representative values from these saved grids for display; no clinical action thresholds were estimated.'),
 ('EBM explanations','Global EBM importance uses the fitted training-distribution weights; the ten terms with the highest across-fold mean native importance are displayed for each outcome, with all five fold values. Response functions were selected for importance, observed-value support, readable clinical quantities and a supported waveform example; they are not restricted to the global top ten. Curves show native additive log-odds differences from labeled reference values before recalibration, with finite-value support from the first outer training fold. Additional functions and missing-value support are in Figure S9 and Table S6. Patient examples in Figure S10 were selected without consulting outcomes or contributions, nearest the 10th and 90th percentiles of pooled combined-EBM risk within each outcome, with case-ID tie-breaking. Each case uses its held-out fold model and calibrator. If f(x) = b + Σtⱼ(xⱼ) and calibrated risk is sigmoid(a + cf(x)), local calibrated-logit contributions are ctⱼ, with baseline a + cb. Grouped signed terms and remaining terms retain the exact sum. Full performance, surgical case mix and outcome ascertainment are in Tables S2–S7. WBC, white blood cell count; CRP, C-reactive protein.')]

def calibration_note():
 return 'The audit found no outer-test calibration leakage across1,440 saved outer-fold bundles. Training/test patients were disjoint,calibration used outer-training OOF predictions,and saved transformations and training-selected Youden thresholds reproduced. Hyperparameters were selected within full outer-training data before calibration cross-fitting; those training predictions are development data,not independent test evidence. All180 displayed calibration-bin rows reproduce held-out outcomes. Close overall mean predicted risk does not establish excellent local calibration: combined-AKI maximum observed bin gaps are9.18percentage points for XGBoost,8.63for LR and1.66for EBM. These are descriptive ten-bin diagnostics,not uncertainty bounds.'

def spaced(s):
 s=re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',s)
 s=re.sub(r',(?=[A-Za-z])',', ',s)
 return re.sub(r'(\d) (st|nd|rd|th)\b',r'\1\2',s)


def draft():
 df=pd.read_csv(ROOT/'04_DATA/results/main_shared/tables/metrics_summary.csv')
 d=init('Manuscript starting document')
 d.add_heading('Routine intraoperative waveform descriptors for postoperative AKI and ICU admission prediction',0)
 d.add_paragraph('AUTHOR-EDITABLE WORKING DRAFT — revised 3 November 2026. Results follow the current figure and table sequence. Introduction, references, interpretation and author declarations require completion.')
 d.add_heading('Structured abstract — working text',1)
 d.add_paragraph('Objective: Assess the information in routine intraoperative waveform descriptors and their incremental value beyond preoperative clinical predictors for postoperative acute kidney injury (AKI) and intensive care unit (ICU) admission at operation end. Methods: Logistic regression, XGBoost and additive explainable boosting machines (EBMs) were evaluated using patient-grouped nested cross-validation. Whole-case analyses were primary; windowed and broader-ICU analyses were exploratory. Results: The main cohort comprised 3,571 operations in 3,432 patients, including 189 AKI and 834 ICU-admission operations. Clinical-only models outperformed waveform-only models in every core comparison. Combined XGBoost and EBM improved ICU average precision by 0.034 and 0.038, respectively, and AUROC by 0.019 and 0.024; paired intervals excluded zero for these improvements. AKI comparisons were inconsistent: the XGBoost AP difference was 0.042 (95% CI −0.018 to 0.095), while logistic regression worsened and EBM showed no clear gain. Global EBM importance showed substantial reliance on surgical case mix and preoperative variables. Conclusion: Incremental discrimination depended on outcome and model family. These internally validated results support further evaluation of waveform information while leaving clinical benefit and transportability unresolved.')
 d.add_heading('Introduction — author outline',1)
 for text in ['Define the operation-end prediction question: what information can generic descriptors from four widely recorded intraoperative channels provide, alone and beyond preoperative clinical risk?', 'Position the study as an evaluation of information sources and interpretable model behavior. Relate it to the INSPIRE intraoperative-information question, while distinguishing its stage 2/3 endpoint, data representation and cohort.', 'State AKI as the primary endpoint and ICU admission as secondary. Specify the clinical-only, waveform-only and combined comparison; avoid inferring comparative usefulness from raw AP across outcomes.']:
  d.add_paragraph(text)
 d.add_heading('Methods — source-verified working text',1)
 for heading,text in method_blocks():d.add_heading(heading,2);d.add_paragraph(text)
 d.add_heading('Results',1)
 d.add_heading('Participants and clinical context — Figure 1 and Table 1',2)
 d.add_paragraph('The main shared cohort comprised 3,571 operations in 3,432 patients. AKI occurred in 189 operations (5.3%), including 53 stage 2–3 events (1.5% of operations); 834 operations were followed by ICU admission (23.4%). Median operative duration was 145 minutes (IQR 86–210). Preoperative eGFR was below 60 mL/min/1.73 m² in 172 operations (4.8%); eGFR was missing in six. Surgical case mix and ascertainment are detailed in Table S7. Transplantation represented 163 operations and 74 AKI events, providing important context for overall discrimination and model interpretation.')
 d.add_heading('Absolute performance — Table 2',2)
 d.add_paragraph('Clinical-only models exceeded waveform-only models in every whole-case core comparison. Waveform-only AKI AP ranged from 0.157 to 0.177 and AUROC from 0.697 to 0.724; corresponding ICU values were 0.512–0.570 and 0.747–0.797. Whole-case precision–recall and ROC curves are provided in Figures S1 and S2. Complete metrics for all input sets are in Tables S2 and S3.')
 d.add_heading('Incremental discrimination — Figure 2',2)
 for outcome,label in [('any_aki','AKI'),('icu_admission','ICU admission')]:
  d.add_paragraph(label+':')
  for family in ['Logistic regression','XGBoost','EBM']:
   row=df[df.outcome.eq(outcome)&df.branch.eq('non_windowed')&df.model_name.eq(family)&df.feature_set.eq('preop_and_all_waveforms')].iloc[0]
   fmt=lambda key:f'{row[key]:+.3f} ({row[key+"_lower"]:+.3f} to {row[key+"_upper"]:+.3f})'
   d.add_paragraph(f'{family}: combined-minus-clinical AP {fmt("delta_auprc")}; AUROC {fmt("delta_auroc")}.')
 d.add_paragraph('Values above use the 0–1 metric scale and 95% paired patient-cluster bootstrap intervals. Figure 2 expresses the same differences in percentage points. The clearest gains occurred in ICU XGBoost and EBM; AKI gains were inconsistent, including deterioration in logistic regression.')
 d.add_heading('Held-out calibration — Figure 3',2)
 d.add_paragraph('Calibration plots compare clinical-only and combined models using pooled outer-test predictions. Low-risk AKI detail is displayed alongside the full probability range. Local departures from the diagonal remain visible despite broadly aligned marginal risks; the plots should be interpreted with the limited number of AKI events in mind.')
 d.add_heading('Model interpretation — Figures 4 and 5',2)
 d.add_paragraph('Transplantation was the most important native EBM term for AKI in every outer fold; colorectal procedures and major resections were the leading ICU terms. The global importance display places waveform terms in the context of the complete model. Selected response functions show the hemoglobin–AKI, age–ICU and CO₂ time-reversibility–ICU associations. Extended functions and four systematically selected held-out examples are in Figures S9 and S10.')
 d.add_heading('Exploratory analyses',2)
 d.add_paragraph('Single-channel comparisons, windowed models and broader ICU sensitivity analyses are presented in Figures S3–S6. Conventional decision curves and paired net-benefit differences are in Figures S7–S8. The broader ICU cohort overlaps the main population and is a sensitivity analysis. Decision thresholds remain exploratory because a specific intervention policy has not been defined.')
 d.add_heading('Discussion — author outline',1)
 for text in ['Lead with information-source findings: waveform-only models carried signal; incremental value depended on outcome and model. Keep the negative and uncertain AKI results visible.', 'Interpret strong surgical case-mix effects before proposing waveform physiology. Global importance describes the fitted combined model; it does not quantify the incremental performance contribution of individual terms.', 'Explain clinical limitations of the recorded ICU-disposition outcome, measurement-dependent AKI ascertainment, laboratory missingness, low-rate waveform representation and single-system internal validation.', 'Present AP as the established primary comparison metric and AUROC as a prominent complementary measure. Evaluate probability calibration separately; distinguish exploratory net benefit from demonstrated clinical impact.', 'Treat local case explanations as illustrations of fitted predictions. Link additional development to external evaluation, waveform-processing sensitivity and a specified clinical decision rather than adding model families.']:
  d.add_paragraph(text)
 d.add_heading('References and source leads',1)
 for text in ['Do et al. Integration of intraoperative data in interpretable machine learning models to predict postoperative AKI in noncardiac surgery patients. JAMIA Open 2026;9:ooag092. https://doi.org/10.1093/jamiaopen/ooag092', 'Van Calster et al. Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance. Lancet Digital Health 2025;7:100916. https://doi.org/10.1016/j.landig.2025.100916', 'TRIPOD+AI statement. BMJ 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378', '[Authors: assemble and verify the complete literature and data references before submission.]']:
  d.add_paragraph(text)
 d.add_heading('Author-controlled declarations',1)
 for label in ['Authors, affiliations and corresponding author','Ethics and consent','Funding and competing interests','Author contributions','Data/code availability and licenses','Acknowledgment/disclosure of substantive AI assistance']:d.add_paragraph(label+': [author to complete]')
 d.save(AUTHOR/'Manuscript_Starting_Document.docx')
 (AUTHOR/'Manuscript_Starting_Document.txt').write_text('\n\n'.join(p.text for p in d.paragraphs)+'\n')
 audit=init('Calibration leakage and interpretation');audit.add_heading('Calibration leakage and interpretation',0);audit.add_paragraph(spaced(calibration_note()));audit.add_paragraph('Evidence: 00_START_HERE/run_audit/calibration_audit.json and accompanying fold, plotted-bin and diagnostic CSVs.');audit.save(AUTHOR/'Calibration_Audit_Explanation.docx')


def navigation():
 d=init('VitalDB manuscript workspace — open first')
 for block in (ROOT/'00_START_HERE/READ_ME_FIRST.txt').read_text().strip().split('\n\n'):
  lines=block.splitlines()
  if lines[0].isupper():d.add_heading(lines[0].capitalize(),1);d.add_paragraph('\n'.join(lines[1:]))
  elif block.startswith('VitalDB'):d.add_heading(block,0)
  else:d.add_paragraph(block)
 d.save(ROOT/'00_START_HERE/OPEN_ME.docx')

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--refresh-author-material',action='store_true',help='Explicitly regenerate the author working draft and reference documents.')
 args=parser.parse_args()
 # Rebuilding figures/tables must preserve the author's manuscript edits.
 frozen=[AUTHOR/'Frozen_Methods_Reference.docx']
 preserved={p:p.read_bytes() for p in frozen if p.exists() and not args.refresh_author_material}
 methods()
 for path,data in preserved.items():path.write_bytes(data)
 if args.refresh_author_material or not (AUTHOR/'Manuscript_Starting_Document.docx').exists():draft()
 books();navigation()
 for name in ('Manuscript_Starting_Document.docx','Manuscript_Starting_Document.txt','Frozen_Methods_Reference.docx','Calibration_Audit_Explanation.docx','DCA_Full_Scale_Reference_Views.docx','Supplementary_Methods_and_Readiness.docx'):
  old=OUT/name
  if old.exists():
   if not (AUTHOR/name).exists():shutil.move(old,AUTHOR/name)
   else:old.unlink()
 print('Publication documents and separate author material built.')
