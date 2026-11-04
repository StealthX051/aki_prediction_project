"""Editable mentor guide: narrative purpose, display order and deliberate information density."""
from pathlib import Path
import importlib.util,json
from docx.shared import Pt
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('documents',Path(__file__).with_name('build_manuscript_documents.py'))
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
OUT=ROOT/'01_MANUSCRIPT'
STEPS=[
 ('Figure 1 — Who entered the analysis?', 'Show source-to-analysis selection and operation/patient counts. State that predictions use information available by operation end. Keep model-development details in Methods.'),
 ('Table 1 — What clinical population does this represent?', 'Give demographics, renal function, operative exposure, surgical case mix and event burden. The three additions are operative duration, eGFR below 60 and the stage 2–3 subset of AKI; the table remains one portrait page.'),
 ('Table 2 — How strong are the three information sources?', 'Present clinical-only, waveform-only and combined models across all three families and both outcomes. Waveform-only models carry signal but do not exceed clinical-only performance in these core comparisons.'),
 ('Figure 2 — What do waveforms add to clinical predictors?', 'Use the same six combined-minus-clinical contrasts for AP and AUROC. Keep all negative and uncertain estimates. The clearest gains are ICU XGBoost and EBM; AKI findings are inconsistent.'),
 ('Figure 3 — How do the predicted probabilities compare with observed risks?', 'Show held-out calibration for clinical and combined models, with full axes and a low-risk AKI inset. Assess probability reliability separately from discrimination; show the held-out risk distributions beneath each panel. Table S8 reports observed/expected events and bin support.'),
 ('Figure 4 — Which terms does the combined EBM rely on?', 'Show native global importance before selected functions. Preserve prominent surgical terms: transplantation for AKI, and colorectal procedures/major resection for ICU. Magnitude does not specify the direction of association.'),
 ('Figure 5 — What do selected fitted functions look like?', 'Show hemoglobin–AKI, age–ICU and CO₂ time reversibility–ICU, with finite training support. These are selected examples, including a supported waveform descriptor; they are not all top-ten terms.')]

def main():
 d=D.init('Figure and table plan')
 d.add_heading('Figure and table plan',0)
 d.add_paragraph('Mentor discussion copy · revised 4 November 2026')
 d.add_heading('Central question',1)
 d.add_paragraph('Do generic descriptors from four routine intraoperative waveforms contain postoperative risk information, and do they add value beyond preoperative clinical predictors at operation end?')
 d.add_paragraph('The supported account is outcome- and model-dependent incremental value. AKI remains the primary endpoint and ICU admission the secondary endpoint. The paper should retain negative AKI results and explain the role of case mix alongside the positive ICU comparisons.')
 d.add_heading('Main reading order',1)
 for heading,text in STEPS:
  p=d.add_paragraph();p.add_run(heading+' ').bold=True;p.add_run(text)
 d.add_page_break()
 d.add_heading('Table 1 and display choices',0)
 d.add_paragraph('Use five main figures and two tables. Preserve legible native-size labels rather than compressing every result into the main paper. The review book interleaves figures and tables in the reading order above; its seven pages preserve portrait Table 1 and landscape Table 2.')
 for label,value,reason in [
  ('Operative duration','145 min (IQR 86–210)','Documents the duration of waveform exposure available to the whole-case descriptors.'),
  ('Preoperative eGFR <60','172 operations (4.8%); six missing','Adds distributional renal-risk context that the median eGFR alone does not show. This is a descriptive preoperative category.'),
  ('Stage 2–3 AKI','53 operations (1.5%)','Shows the severity composition of the primary any-stage AKI outcome.')]:
  p=d.add_paragraph();p.add_run(label+': '+value+'. ').bold=True;p.add_run(reason)
 d.add_paragraph('The existing age, BMI, sex, ASA, emergency/open surgery, hypertension/diabetes, procedure categories and preoperative laboratory values are sufficient for the compact main table. WBC and CRP missingness remain visible because they help interpret the fitted models. Extensive additional comorbidities, univariate p-values and fold-by-fold demographics would add little to this account.')
 d.add_paragraph('Table S7 supplies all surgical categories with within-category AKI/ICU frequencies, full creatinine-defined AKI stages, repeat-patient counts and 48-/72-hour laboratory availability. It gives the selection and case-mix detail needed to interpret transportability without widening Table 1.')
 d.add_heading('Metric presentation',1)
 d.add_paragraph('Retain the established AP analysis hierarchy and make AUROC visually prominent. AP summarizes the precision–recall tradeoff; AUROC measures ranking discrimination. Neither is a measure of clinical benefit. Main conclusions should reflect agreement or disagreement across AP, AUROC, calibration and the exploratory decision curves. Avoid direct comparisons of raw AP between AKI and ICU, whose event prevalences differ.')
 d.add_heading('Explainability choices',1)
 d.add_paragraph('The global native-importance figure answers which terms matter before the selected function figure shows their shapes. All folds are visible. Additional waveform functions remain supplementary when their interpretation or support is less secure. Remove standalone missing-bin dotplots; preserve their numerical support in Table S6 and the full electronic archive.')
 d.add_paragraph('Figure S10 illustrates native additive reasoning in four held-out operations selected by predicted-risk percentiles without consulting outcomes. The displayed clinical and combined risks come from separate fitted models. Channel contributions are signed sums within the combined EBM; they do not equal the difference between the two models’ predicted risks.')
 d.add_page_break()
 d.add_heading('Supplementary displays',0)
 _,supp,_=D.entries()
 PURPOSES={1:'Absolute precision–recall behavior of the complete whole-case core comparison.',2:'Familiar ROC view of the same held-out predictions and AUROCs.',3:'Single-/multichannel waveform-only AP contrasts against clinical-only models.',4:'The corresponding waveform-only AUROC contrasts.',5:'Exploratory windowed combined-versus-clinical AP differences.',6:'Broader ICU cohort sensitivity; this overlapping cohort is not an external validation.',7:'Conventional clinical and combined decision curves with treat-all/treat-none.',8:'Paired combined-minus-clinical net-benefit differences and pointwise intervals.',9:'Additional AWP, WBC and processed PLETH functions with support.',10:'Four exact held-out EBM case decompositions; illustrative rather than an accuracy claim.'}
 for i,f in enumerate(supp,1):
  p=d.add_paragraph();p.add_run(f['label']+' — ').bold=True;p.add_run(PURPOSES[i])
 d.add_heading('Supplementary tables',1)
 for text in ['S1: complete clinical descriptives and missingness for both cohorts.', 'S2/S2B: main-cohort performance and paired differences; the input-set key lists the predictors.', 'S3/S3B: all 90 broader ICU configurations and paired differences.', 'S4: matched model-family comparisons.', 'S5: representative saved-grid decision-curve values; the complete grid remains electronic.', 'S6: fold-level importance and missing-value support for the six displayed response terms.', 'S7: surgical case mix, renal context, AKI severity and outcome ascertainment.', 'S8: held-out calibration diagnostics for the clinical and combined models.', 'S9: waveform descriptors displayed in the EBM figures.']:
  d.add_paragraph(text)
 d.add_heading('Source guidance',1)
 for text in ['Journal of Medical Systems: https://link.springer.com/journal/10916/submission-guidelines', 'TRIPOD+AI: https://www.bmj.com/content/385/bmj-2023-078378', 'INSPIRE AKI display precedent: https://doi.org/10.1093/jamiaopen/ooag092', 'Clinical performance measures: https://doi.org/10.1016/j.landig.2025.100916']:
  d.add_paragraph(text)
 d.save(OUT/'Figure_Table_Story_Map.docx')
 (OUT/'Figure_Table_Story_Map.txt').write_text('\n\n'.join(p.text for p in d.paragraphs)+'\n')
 payload={'reading_order':[h for h,_ in STEPS],'main_figures':5,'main_tables':2,'supplementary_figures':len(supp),'main_table_additions':['operative_duration','egfr_below_60','stage_2_3_aki'],'missing_bin_dotplots_published':False}
 (OUT/'story_map.json').write_text(json.dumps(payload,indent=2)+'\n')
 print('Editable narrative guide built')
if __name__=='__main__':main()
