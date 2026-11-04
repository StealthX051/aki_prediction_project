"""Portable OOXML Excel workbooks and complete native Word electronic tables.

Uses only Python's standard library plus the table rebuild's pandas/python-docx.
Strings are stored as text, not formulas; numerical source columns stay numeric.
"""
from pathlib import Path
import json,math,numbers,zipfile,xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from rendering_support.table_assets import numeric_display_columns
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_CODE'))
from publication_style import apply_document_style, WORD_TABLE_PT, WORD_FONT, color

def colname(index):
    value=''
    while index:
        index,r=divmod(index-1,26);value=chr(65+r)+value
    return value


def cell_xml(row,col,value,style=0):
    ref=colname(col)+str(row)
    if value is None or pd.isna(value):return f'<c r="{ref}" s="{style}"/>'
    if isinstance(value,bool):return f'<c r="{ref}" s="{style}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value,numbers.Real):
        if not math.isfinite(float(value)):return f'<c r="{ref}" s="{style}"/>'
        return f'<c r="{ref}" s="{style}"><v>{repr(int(value)) if isinstance(value,numbers.Integral) else repr(float(value))}</v></c>'
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(str(value))}</t></is></c>'


def reading_bold_cells(df):
    """Structural hierarchy and displayed-precision point-estimate emphasis."""
    bold=set()
    groups=[c for c in ('Outcome','Branch','Model') if c in df]
    metrics=[c for c in df if c in {'AP','AUPRC','AUROC','Brier','Brier Score','Sensitivity','Specificity','Precision','F1','F1 Score','Accuracy'}]
    if metrics:
        grouped=df.groupby(groups,sort=False,dropna=False) if groups else [(None,df)]
        for _,frame in grouped:
            for column in metrics:
                values=frame[column].map(lambda value:float(str(value).split(' (',1)[0].replace('−','-')) if str(value).strip() else float('nan'))
                values=values.map(lambda value:float(f'{value:.3f}'))
                best=values.min() if column in {'Brier','Brier Score'} else values.max()
                for index in frame.index:
                    if pd.notna(values.loc[index]) and values.loc[index]==best:
                        bold.add((df.index.get_loc(index),df.columns.get_loc(column)))
        for pos in range(len(df)):
            if pos==0 or any(df.iloc[pos][c]!=df.iloc[pos-1][c] for c in groups):
                for c in ('Outcome','Model'):
                    if c in df:bold.add((pos,df.columns.get_loc(c)))
    if 'Characteristic' in df:
        headings={'Study population','Demographics','Operative characteristics','Comorbidities','Procedure type','Preoperative laboratory values'}
        for pos in range(len(df)):
            if df.iloc[pos]['Characteristic'] in headings:
                bold.update((pos,col) for col in range(len(df.columns)))
    return bold


def write_xlsx(path,sheets):
    """Write freeze panes, filterable columns, print titles, and native typed values."""
    names=[x[0] for x in sheets];assert len(names)==len(set(names)) and all(len(x)<=31 for x in names)
    content=['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>','<Default Extension="xml" ContentType="application/xml"/>','<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>','<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    for i in range(1,len(sheets)+1):content.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    styles=f'''<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="{NS}"><numFmts count="4"><numFmt numFmtId="164" formatCode="0.000000"/><numFmt numFmtId="165" formatCode="0.0"/><numFmt numFmtId="166" formatCode="0.0000"/><numFmt numFmtId="167" formatCode="0.000"/></numFmts><fonts count="3"><font><sz val="11"/><name val="{WORD_FONT}"/></font><font><b/><sz val="11"/><name val="{WORD_FONT}"/></font><font><b/><sz val="14"/><name val="{WORD_FONT}"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF{color('table_header').lstrip('#')}"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left/><right/><top style="thin"><color rgb="FF{color('reference').lstrip('#')}"/></top><bottom style="thin"><color rgb="FF{color('reference').lstrip('#')}"/></bottom><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="12"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0"><alignment vertical="center" wrapText="1"/></xf><xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"><alignment horizontal="right" vertical="top"/></xf><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment horizontal="right" vertical="top" wrapText="1"/></xf><xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"><alignment horizontal="right" vertical="top"/></xf><xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"><alignment horizontal="right" vertical="top"/></xf><xf numFmtId="167" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"><alignment horizontal="right" vertical="top"/></xf><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"><alignment horizontal="right" vertical="top" wrapText="1"/></xf><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="164" fontId="1" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"><alignment horizontal="right" vertical="top"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    workbook=f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><bookViews><workbookView/></bookViews><sheets>'
    workbook+=''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i,name in enumerate(names,1))+'</sheets><definedNames>'
    workbook+=''.join(f'<definedName name="_xlnm.Print_Titles" localSheetId="{i}">\'{escape(name)}\'!$4:$4</definedName>' for i,name in enumerate(names))+'</definedNames><calcPr calcId="191029"/></workbook>'
    rels='<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1,len(sheets)+1))+f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'+''.join(content)+'</Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml',workbook);z.writestr('xl/_rels/workbook.xml.rels',rels);z.writestr('xl/styles.xml',styles)
        for i,(name,title,notes,df,raw) in enumerate(sheets,1):
            lastcol=colname(len(df.columns));lastrow=len(df)+4;numeric=numeric_display_columns(df)
            emphasis=reading_bold_cells(df) if not raw else set()
            parts=[f'<worksheet xmlns="{NS}"><dimension ref="A1:{lastcol}{lastrow}"/><sheetViews><sheetView workbookViewId="0" zoomScale="90"><pane ySplit="4" topLeftCell="A5" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="20"/><cols>']
            column_widths=[]
            for j,c in enumerate(df.columns,1):
                label=str(c).lower();w=42 if ('feature' in label or 'inputs' in label or 'characteristic' in label) else 24 if 'model' in label or label in {'outcome','cohort'} else 18 if c in numeric else 23
                if raw:w=min(32,max(14,len(str(c))+2))
                column_widths.append(w)
                parts.append(f'<col min="{j}" max="{j}" width="{w}" customWidth="1"/>')
            note_height=max(32,math.ceil(len(notes)/max(20,sum(column_widths)*.9))*14+4)
            parts+=['</cols><sheetData>','<row r="1" ht="24" customHeight="1">'+cell_xml(1,1,title,2)+'</row>',f'<row r="2" ht="{note_height}" customHeight="1">'+cell_xml(2,1,notes,3)+'</row>','<row r="4" ht="32" customHeight="1">'+''.join(cell_xml(4,j,c,1) for j,c in enumerate(df.columns,1))+'</row>']
            for rowno,row in enumerate(df.itertuples(index=False,name=None),5):
                line_count=max(sum(max(1,math.ceil(len(line)/max(8,width*.9))) for line in str(value).replace(' (','\n(').splitlines()) for value,width in zip(row,column_widths)) if not raw else 1
                row_height=max(20,min(100,line_count*14+4))
                parts.append(f'<row r="{rowno}"'+(f' ht="{row_height}" customHeight="1"' if not raw else '')+'>')
                for j,(c,v) in enumerate(zip(df.columns,row),1):
                    style=4 if isinstance(v,numbers.Real) and not isinstance(v,numbers.Integral) else 5 if c in numeric else 0
                    if not raw and style==4:
                        if c=='Threshold (%)':style=6
                        elif c in {'Missing-bin score','Calibration slope'}:style=7
                        elif 'Treat-none' in c:style=8
                    if not raw and (rowno-5,j-1) in emphasis:style=11 if style==4 else 9 if c in numeric else 10
                    if not raw and isinstance(v,str) and ' (' in v:v=v.replace(' (','\n(')
                    parts.append(cell_xml(rowno,j,v,style))
                parts.append('</row>')
            parts.append(f'</sheetData><autoFilter ref="A4:{lastcol}{lastrow}"/><mergeCells count="2"><mergeCell ref="A1:{lastcol}1"/><mergeCell ref="A2:{lastcol}2"/></mergeCells><printOptions horizontalCentered="1"/><pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" header="0.2" footer="0.2"/><pageSetup paperSize="9" orientation="landscape" fitToWidth="1" fitToHeight="0"/></worksheet>')
            z.writestr(f'xl/worksheets/sheet{i}.xml',''.join(parts))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name in z.namelist():ET.fromstring(z.read(name))


def fast_docx(path,title,notes,df):
    """Native editable Word table, built linearly for the complete long supplements."""
    doc=apply_document_style(Document(), body_size=WORD_TABLE_PT);sec=doc.sections[0];sec.page_width=Inches(297/25.4);sec.page_height=Inches(210/25.4)
    sec.top_margin=sec.bottom_margin=Inches(.45);sec.left_margin=sec.right_margin=Inches(.45)
    doc.add_heading(title,1);doc.add_paragraph(notes)
    from publication_navigation import word_section_index,bookmark
    from docx.text.paragraph import Paragraph
    context_columns=[c for c in ['Cohort','Outcome','Branch'] if c in df]
    starts={};headings=[];previous=None
    if context_columns:
        for position,values in enumerate(df[context_columns].astype(str).itertuples(index=False,name=None)):
            if values!=previous:
                starts[position]=len(headings);headings.append(' — '.join(values));previous=values
    word_section_index(doc,headings)
    numeric=numeric_display_columns(df)
    weights=[3 if c in {'Feature','Variable','Feature set'} else 2 if ('inputs' in c.lower() or c in {'Cohort','Outcome','Clinical section'}) else 1.3 for c in df]
    available=int((sec.page_width-sec.left_margin-sec.right_margin)/635)
    widths=[int(available*w/sum(weights)) for w in weights]
    tbl=OxmlElement('w:tbl');pr=OxmlElement('w:tblPr');layout=OxmlElement('w:tblLayout');layout.set(qn('w:type'),'fixed');pr.append(layout)
    borders=OxmlElement('w:tblBorders')
    for edge in ('top','bottom','insideH','left','right','insideV'):
        e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single' if edge in {'top','bottom','insideH'} else 'nil');e.set(qn('w:sz'),'2');e.set(qn('w:color'),color('table_rule').lstrip('#'));borders.append(e)
    pr.append(borders);tbl.append(pr);grid=OxmlElement('w:tblGrid')
    for w in widths:e=OxmlElement('w:gridCol');e.set(qn('w:w'),str(w));grid.append(e)
    tbl.append(grid)
    for rowno,row in enumerate([list(df.columns),*df.fillna('').astype(str).values.tolist()]):
        tr=OxmlElement('w:tr');trpr=OxmlElement('w:trPr');trpr.append(OxmlElement('w:cantSplit'))
        if rowno==0:trpr.append(OxmlElement('w:tblHeader'))
        tr.append(trpr)
        for c,value,width in zip(df.columns,row,widths):
            tc=OxmlElement('w:tc');tcpr=OxmlElement('w:tcPr');cw=OxmlElement('w:tcW');cw.set(qn('w:w'),str(width));cw.set(qn('w:type'),'dxa');tcpr.append(cw)
            if rowno==0:sh=OxmlElement('w:shd');sh.set(qn('w:fill'),color('table_header').lstrip('#'));tcpr.append(sh)
            tc.append(tcpr);p=OxmlElement('w:p');pp=OxmlElement('w:pPr');spacing=OxmlElement('w:spacing')
            for k,v in {'before':'0','after':'0','line':'240','lineRule':'auto'}.items():spacing.set(qn('w:'+k),v)
            pp.append(spacing);jc=OxmlElement('w:jc');jc.set(qn('w:val'),'right' if c in numeric else 'left');pp.append(jc);p.append(pp)
            r=OxmlElement('w:r');rp=OxmlElement('w:rPr');sz=OxmlElement('w:sz');sz.set(qn('w:val'),str(round(2*WORD_TABLE_PT)));rp.append(sz)
            fonts=OxmlElement('w:rFonts');fonts.set(qn('w:ascii'),WORD_FONT);fonts.set(qn('w:hAnsi'),WORD_FONT);rp.append(fonts)
            if rowno==0:rp.append(OxmlElement('w:b'))
            r.append(rp);t=OxmlElement('w:t');t.set(qn('xml:space'),'preserve');t.text=str(value);r.append(t);p.append(r)
            if rowno-1 in starts and c==df.columns[0]:
                number=starts[rowno-1];bookmark(Paragraph(p,doc),f'TableSection_{number}',number+1)
            tc.append(p);tr.append(tc)
        tbl.append(tr)
    doc._body._body.insert(len(doc._body._body)-1,tbl);doc.save(path)


def build_workbooks(here):
    here=Path(here)
    notes='Estimates include 95% CIs. Bold identifies the highest displayed estimate within each outcome, branch and model (lowest for Brier).'
    readme=pd.DataFrame([['Prepared','3 November 2026'],['Metric abbreviations','AP, average precision; AUROC, area under the receiver operating characteristic curve; LR, logistic regression.'],['Source preservation','All original numerical CSV files are included in inputs, unchanged.'],['Intervals','Nominal patient-cluster bootstrap intervals conditional on saved fitted predictions.'],['Thresholds','Classification uses saved fold-specific, training-selected thresholds.'],['AP difference','Tables use the 0–1 AP scale; forest figures use percentage points.'],['Mortality','Exploratory meeting backup; source linkage unresolved.']],columns=['Item','Definition'])
    reading=[('READ_ME','Table guide',notes,readme,False)]
    manifest=[]
    choices=[('main/Table_1_cohort_display.csv','Table1_Main'),('main/Table_2_core_comparisons.csv','Table2_Main'),('supplement/Table_S2_input_key.csv','S2_Input_Key'),('supplement/Table_S2_core_all_metrics_display.csv','S2_Core'),('supplement/Table_S2_main_all_configurations_display.csv','S2_Main_All'),('supplement/Table_S2B_main_paired_differences_display.csv','S2B_Main_Differences'),('supplement/Table_S3_broader_ICU_all_configurations_display.csv','S3_Broader_ICU'),('supplement/Table_S3B_broader_ICU_paired_differences_display.csv','S3B_ICU_Differences'),('supplement/Table_S1_descriptives_missingness_display.csv','S1_Descriptives'),('supplement/Table_S4_family_comparisons_display.csv','S4_Family_Comparisons'),('supplement/Table_S5A_absolute_net_benefit_display.csv','S5A_Absolute_DCA'),('supplement/Table_S5_paired_decision_curve_values_display.csv','S5_Decision_Curves'),('supplement/Table_S6_EBM_fold_importance_display.csv','S6_EBM_Importance'),('supplement/Table_S6B_selected_missing_bins_display.csv','S6B_Missing_Bins'),('meeting_backup/Meeting_backup_mortality_all_metrics_display.csv','Mortality_Backup')]
    for rel,sheet in choices:
        p=here/rel;df=pd.read_csv(p,keep_default_na=False);title=sheet.replace('_',' ')
        # Reuse the table title/key rather than maintain a second wording dictionary.
        md_path=p.with_name(p.stem.removesuffix('_display')+'.md')
        blocks=md_path.read_text().strip().split('\n\n')
        title=next(block.removeprefix('# ').strip() for block in blocks if block.startswith('# '))
        sheet_notes=next((block.strip() for block in blocks if block.strip() and not block.startswith(('#','|','_No rows'))),notes)
        if sheet=='Mortality_Backup':sheet_notes='52 outcome-positive operations from 47 patients. '+sheet_notes
        reading.append((sheet,title,sheet_notes,df,False));manifest.append({'file':rel,'worksheet':sheet,'rows':len(df),'columns':len(df.columns)})
        if sheet in {'S1_Descriptives','S4_Family_Comparisons','S5A_Absolute_DCA','S5_Decision_Curves','S6_EBM_Importance','S6B_Missing_Bins'}:
            fast_docx(p.with_name(p.stem.removesuffix('_display')+'.docx'),title,sheet_notes,df)
    raw=[('READ_ME','Complete numeric sources','Values retain source precision; identifiers are text. CSVs are authoritative.',readme,False)]
    input_choices=[('source_main_shared_metrics_summary.csv','Main_180'),('source_icu_waveform_sensitivity_metrics_summary.csv','Broader_ICU_90'),('source_mortality_waveform_sensitivity_metrics_summary.csv','Mortality_18'),('Table_1_cohort.csv','Cohort_Source'),('Table_S1_descriptives_missingness.csv','S1_Complete'),('Table_S4_family_comparisons.csv','S4_Comparisons'),('Table_S5_paired_decision_curve_values.csv','S5_Paired_DCA'),('Table_S6_EBM_fold_importance.csv','S6_EBM_Terms'),('Table_S6_selected_missing_bins.csv','S6_Missing_Bins')]
    raw_titles={'Main_180':'Main cohort model results','Broader_ICU_90':'Broader ICU model results','Mortality_18':'Exploratory mortality results','Cohort_Source':'Cohort characteristics','S1_Complete':'Cohort characteristics and missingness','S4_Comparisons':'Paired model comparisons','S5_Paired_DCA':'Paired net-benefit differences','S6_EBM_Terms':'EBM term importance','S6_Missing_Bins':'EBM missing-value scores and support'}
    for filename,sheet in input_choices:
        df=pd.read_csv(here/'inputs'/filename);raw.append((sheet,raw_titles[sheet],'Source values; definitions are in the accompanying documentation.',df,True))
    for scope,short in [('main_shared','Main'),('icu_waveform_sensitivity','ICU'),('mortality_waveform_sensitivity','Mortality')]:
        for kind,label in [('summary','Absolute'),('comparisons','Paired')]:
            source=here/'inputs'/f'{scope}__decision_curve_{kind}.csv'
            if source.exists():
                df=pd.read_csv(source)
                raw.append((f'DCA_{short}_{label}',('Net benefit: ' if kind=='summary' else 'Paired net benefit: ')+{'Main':'main cohort','ICU':'broader ICU cohort','Mortality':'exploratory mortality'}[short],'Net benefit per operation; threshold probabilities use the 0–1 scale.',df,True))
    write_xlsx(here/'workbooks/Manuscript_Tables_Editable.xlsx',reading)
    write_xlsx(here/'workbooks/Complete_Numeric_Sources.xlsx',raw)
    (here/'qa/workbook_manifest.json').write_text(json.dumps({'reading_sheets':manifest,'numeric_sheets':[{'worksheet':s[0],'rows':len(s[3]),'columns':len(s[3].columns)} for s in raw]},indent=2)+'\n')
    print('Created editable reading and numeric workbooks; complete electronic Word tables.')
