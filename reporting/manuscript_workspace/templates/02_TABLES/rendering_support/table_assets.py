# Standalone subset adapted from the project manuscript table exporters.
from __future__ import annotations
from datetime import datetime, timezone
from html import unescape
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '05_CODE'))
from publication_navigation import bookmark, word_section_index, table_footer, add_pdf_navigation
from publication_style import (apply_document_style, WORD_FONT, WORD_TABLE_PT, FONT_FAMILY, color, style_context)

def _clean_cell(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unescape(text)
    return text.replace("&nbsp;", " ").replace("\n", "<br>")

def dataframe_to_markdown(df: pd.DataFrame, styles=None) -> str:
    """Render a dataframe as a plain Markdown table without optional deps."""

    rows = [[("**" + _clean_cell(value) + "**")
             if _clean_cell(value).strip() and (styles or {}).get(index, {}).get(column, {}).get("bold")
             else _clean_cell(value)
             for column, value in zip(df.columns, row)]
            for index, row in zip(df.index, df.itertuples(index=False, name=None))]
    headers = [_clean_cell(column) for column in df.columns]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def _format_row(cells: Sequence[str]) -> str:
        padded = [cell.ljust(widths[idx]) for idx, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)

def numeric_display_columns(display: pd.DataFrame) -> set:
    """Recognize supplied numeric/interval strings while retaining text identities."""
    identities = {"model", "model_name", "outcome", "branch", "feature set", "feature_set",
                  "input set", "feature set key", "pipeline", "inputs", "inputs text", "caseid", "subjectid"}
    metric_labels = {"ap", "auprc", "auroc", "brier", "brier score", "sensitivity",
                     "specificity", "precision", "f1", "f1 score", "accuracy"}
    result = set()
    for column in display.columns:
        label = str(column).lower().removeprefix("δ ").strip()
        values = [str(value).strip() for value in display[column].dropna()
                  if str(value).strip() not in {"", "—", "Reference"}]
        prepared = bool(values) and all(re.fullmatch(
            r"[+−-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+−-]?\d+)?%?(?:\s*\(.*\))?", value)
            for value in (v.replace(",", "") for v in values))
        if label not in identities and (label in metric_labels or pd.api.types.is_numeric_dtype(display[column]) or prepared):
            result.add(column)
    return result

def symbol_display_columns(display: pd.DataFrame) -> set:
    return {column for column in display if set(display[column].dropna().astype(str)) <= {"✓", "—", ""}
            and display[column].astype(str).eq("✓").any()}

def _add_dataframe_to_doc(document: Document, df: pd.DataFrame, styles=None) -> None:
    table = document.add_table(rows=1 + len(df), cols=len(df.columns))
    table.style = None
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH", "left", "right", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single" if edge in {"top", "bottom", "insideH"} else "nil")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), color("table_rule").lstrip("#"))
        borders.append(border)
    table._tbl.tblPr.append(borders)
    table.autofit = False
    repeat_header = OxmlElement("w:tblHeader")
    table.rows[0]._tr.get_or_add_trPr().append(repeat_header)
    for row in table.rows:
        keep_row = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(keep_row)
    widths = _table_widths(df)
    available = document.sections[-1].page_width - document.sections[-1].left_margin - document.sections[-1].right_margin

    numeric_columns = numeric_display_columns(df)
    symbol_columns = symbol_display_columns(df)
    header_cells = table.rows[0].cells
    for idx, column in enumerate(df.columns):
        header_cells[idx].width = int(available * widths[idx])
        table.columns[idx].width = int(available * widths[idx])
        header_cells[idx].text = str(column)
        if header_cells[idx].paragraphs and header_cells[idx].paragraphs[0].runs:
            header_run = header_cells[idx].paragraphs[0].runs[0]
            header_run.bold = True
            header_run.font.name = WORD_FONT
            header_run.font.size = Pt(WORD_TABLE_PT)
        header_cells[idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if column in symbol_columns else WD_ALIGN_PARAGRAPH.RIGHT if column in numeric_columns else WD_ALIGN_PARAGRAPH.LEFT

    for row_idx, row in enumerate(df.itertuples(index=False, name=None), start=1):
        cells = table.rows[row_idx].cells
        for col_idx, value in enumerate(row):
            cells[col_idx].width = int(available * widths[col_idx])
            cells[col_idx].text = _clean_cell(value).replace("<br>", "\n")
            cells[col_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if df.columns[col_idx] in symbol_columns else WD_ALIGN_PARAGRAPH.RIGHT if df.columns[col_idx] in numeric_columns else WD_ALIGN_PARAGRAPH.LEFT
            style = (styles or {}).get(df.index[row_idx - 1], {}).get(df.columns[col_idx], {})
            if style.get("top_rule"):
                cell_borders = OxmlElement("w:tcBorders")
                top = OxmlElement("w:top")
                for key, value in {"val": "single", "sz": "6", "color": color("reference").lstrip("#")}.items():
                    top.set(qn("w:" + key), value)
                cell_borders.append(top)
                cells[col_idx]._tc.get_or_add_tcPr().append(cell_borders)
            if style.get("bg"):
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), style["bg"].lstrip("#"))
                cells[col_idx]._tc.get_or_add_tcPr().append(shading)
            for paragraph in cells[col_idx].paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1
                for run in paragraph.runs:
                    run.font.name = WORD_FONT
                    run.font.size = Pt(WORD_TABLE_PT)
                    run.bold = bool(style.get("bold"))

def _table_widths(df):
    weights = [max(10, min(30, max([len(str(c))] + [len(str(v).split(" (")[0]) for v in df[c]]))) for c in df]
    return [weight / sum(weights) for weight in weights]

def write_markdown_sections(
    path: Path,
    *,
    title: str,
    sections: Sequence[Mapping[str, object]],
) -> Path:
    """Write a Markdown document containing one or more table sections."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for section in sections:
        heading = str(section["heading"])
        df = section.get("dataframe")
        text = section.get("text")
        if heading != title:
            lines.extend([f"## {heading}", ""])
        if text:
            lines.append(str(text))
            lines.append("")
        if isinstance(df, pd.DataFrame) and not df.empty:
            lines.append(dataframe_to_markdown(df, section.get("styles")))
            lines.append("")
        elif isinstance(df, pd.DataFrame):
            lines.append("_No rows._")
            lines.append("")
        if section.get("notes"):
            lines.extend([str(section["notes"]), ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path

def write_docx_sections(
    path: Path,
    *,
    title: str,
    sections: Sequence[Mapping[str, object]],
) -> Path:
    """Write a DOCX document containing one or more table sections."""

    path.parent.mkdir(parents=True, exist_ok=True)
    document = apply_document_style(Document(), body_size=WORD_TABLE_PT)
    for section in document.sections:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = section.page_height, section.page_width

    document.add_heading(title, level=0)
    navigable=[section for section in sections if isinstance(section.get('dataframe'),pd.DataFrame)]
    headings=[str(section['heading']) for section in navigable]
    word_section_index(document,headings)
    section_number=0
    for section in sections:
        heading_paragraph=None
        if str(section["heading"]) != title:
            heading_paragraph=document.add_heading(str(section["heading"]), level=1)
        if isinstance(section.get('dataframe'),pd.DataFrame):
            if heading_paragraph is None:heading_paragraph=document.paragraphs[0]
            bookmark(heading_paragraph,f'TableSection_{section_number}',section_number+1)
            section_number+=1
        text = section.get("text")
        if text:
            document.add_paragraph(str(text))
        df = section.get("dataframe")
        if isinstance(df, pd.DataFrame):
            _add_dataframe_to_doc(document, df, section.get("styles"))
        if section.get("notes"):
            document.add_paragraph(str(section["notes"]))
        document.add_paragraph()
    document.save(path)
    return path

def write_pdf_sections(
    path: Path,
    *,
    title: str,
    sections: Sequence[Mapping[str, object]],
) -> Path:
    """Write a PDF document containing one or more table sections."""

    path.parent.mkdir(parents=True, exist_ok=True)
    headings=[str(section['heading']) for section in sections]
    section_pages=[];links=[]
    with style_context(), PdfPages(path) as pdf:
        pdf._publication_navigation={'headings':headings,'links':links}
        for section_number,section in enumerate(sections):
            pdf._publication_navigation['current']=section_number
            section_pages.append((str(section['heading']),pdf.get_pagecount()+1))
            heading = str(section["heading"])
            df = section.get("dataframe")
            text = section.get("text")
            title_text = str(section.get("pdf_title") or (title if heading == title else f"{title} | {heading}"))
            if text and not isinstance(df, pd.DataFrame):
                fig, ax = plt.subplots(figsize=(12, 3.5))
                ax.axis("off")
                ax.set_title(title_text, fontsize=14, fontweight="bold", pad=18)
                ax.text(0.02, 0.95, str(text), va="top", ha="left", wrap=True)
                pdf.savefig(fig, bbox_inches="tight", pad_inches=.2)
                plt.close(fig)
                continue
            if isinstance(df, pd.DataFrame):
                _render_pdf_table_page(pdf, title_text, df, notes=section.get("notes"), styles=section.get("styles"))
    if section_pages:
        record=add_pdf_navigation(path,title,section_pages,links)
        # Sidecars keep link destinations inspectable without a GUI dependency.
        path.with_suffix('.navigation.json').write_text(json.dumps(record,indent=2)+'\n')
    return path

def dataframe_to_html(
    display: pd.DataFrame, *, title: str, notes: Optional[str] = None,
    styles: Optional[dict] = None,
) -> str:
    """Render one styled reading table without creating duplicate format files."""
    # Preserve prepared display strings and raw numeric precision consistently
    # with Markdown; pandas' default HTML formatter otherwise rounds to 6 places.
    styler = display.style.hide(axis="index").format(
        lambda value: "" if pd.isna(value) else str(value), escape="html")
    numeric_columns = numeric_display_columns(display)
    symbol_columns = symbol_display_columns(display)
    column_styles = []
    for position, column in enumerate(display.columns):
        numeric = column in numeric_columns
        props = [("text-align", "center" if column in symbol_columns else "right" if numeric else "left")]
        if numeric:
            props.append(("font-variant-numeric", "tabular-nums"))
        column_styles.append({"selector": f"th.col_heading.col{position}, td.col{position}", "props": props})
    styler.set_table_styles(column_styles)
    if styles:
        styler = styler.apply(lambda column: [
            ("background-color:" + styles.get(i, {}).get(column.name, {}).get("bg", "white") + ";")
            + ("font-weight:bold;" if styles.get(i, {}).get(column.name, {}).get("bold") else "")
            + ("border-top:1px solid " + color("reference") + ";" if styles.get(i, {}).get(column.name, {}).get("top_rule") else "")
            for i in column.index], axis=0)
    from html import escape
    html = '<!doctype html><meta charset="utf-8"><style>body{font:11pt Georgia,serif;color:#222;margin:2em;max-width:1100px}h1{font-size:15pt}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #ddd}thead th{border-top:1px solid #777;border-bottom:1px solid #777}p{font-size:9pt;line-height:1.5}@media print{body{margin:0;max-width:none}thead{display:table-header-group}tfoot{display:table-footer-group}tr{break-inside:avoid}th,td{padding:5px 7px}}</style>'
    html = html.replace('font:11pt Georgia,serif', 'font:11pt ' + WORD_FONT + ',sans-serif').replace('color:#222', 'color:' + color('ink')).replace('#ddd', color('table_rule')).replace('#777', color('reference'))
    html += f"<h1>{escape(title)}</h1>" + styler.to_html() + f"<p>{escape(notes or '')}</p>"
    return html

def export_dataframe_bundle(
    base_path: Path,
    *,
    title: str,
    dataframe: pd.DataFrame,
    include_csv: bool = True,
    display_dataframe: Optional[pd.DataFrame] = None,
    notes: Optional[str] = None,
    styles: Optional[dict] = None,
) -> Dict[str, str]:
    """Export a single table to manuscript-friendly formats."""

    base = base_path.with_suffix("") if base_path.suffix else base_path
    display = dataframe if display_dataframe is None else display_dataframe
    sections = [{"heading": title, "dataframe": display, "notes": notes, "styles": styles}]
    outputs: Dict[str, str] = {}
    if include_csv:
        csv_path = base.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(csv_path, index=False)
        outputs["csv"] = str(csv_path)
    md_path = write_markdown_sections(base.with_suffix(".md"), title=title, sections=sections)
    docx_path = write_docx_sections(base.with_suffix(".docx"), title=title, sections=sections)
    pdf_path = write_pdf_sections(base.with_suffix(".pdf"), title=title, sections=sections)
    outputs["md"] = str(md_path)
    outputs["docx"] = str(docx_path)
    outputs["pdf"] = str(pdf_path)
    html = dataframe_to_html(display, title=title, notes=notes, styles=styles)
    base.with_suffix(".html").write_text(html, encoding="utf-8")
    outputs["html"] = str(base.with_suffix(".html"))
    return outputs

def _render_pdf_table_page(pdf, title, df, *, notes=None, styles=None):
    """Fixed A4 landscape pages, readable type and balanced continuation blocks."""
    width,height=297/25.4,210/25.4
    if len(df.columns)>7:
        ids=[c for c in df if str(c).lower() in {'model','outcome','branch','feature set'}]
        ids=ids[:2] or list(df.columns[:1]);values=[c for c in df if c not in ids]
        for start in range(0,len(values),5):
            block=values[start:start+5]
            _render_pdf_table_page(pdf,title+'\n'+block[0]+' to '+block[-1],df[ids+block],notes=notes,styles=styles)
        return
    font=9
    ids=[c for c in df if str(c).lower() in {'model','outcome','branch','feature set'}]
    weights=[2.7 if c=='Feature set' else 1.25 if c=='Model' else 1.55 for c in df]
    fractions=[v/sum(weights) for v in weights]
    available_width=width-.65
    def wrap(value,j):
        value=_clean_cell(value).replace('<br>','\n').replace(' (','\n(')
        n=max(13,int(available_width*fractions[j]*72/(font*.49)))
        return '\n'.join(textwrap.fill(line,n,break_long_words=False) for line in value.splitlines())
    headers=[wrap(c,j) for j,c in enumerate(df.columns)]
    rows=[[wrap(v,j) for j,v in enumerate(row)] for row in df.itertuples(index=False,name=None)]
    title_lines='\n'.join(textwrap.fill(line,110,break_long_words=False) for line in title.splitlines())
    note_lines=textwrap.wrap(str(notes or ''),165)
    title_h=(title_lines.count('\n')+1)*.19+.17
    note_h=len(note_lines)*.135+.12
    head_h=max(v.count('\n')+1 for v in headers)*font*1.18/72+.08
    heights=[max(v.count('\n')+1 for v in row)*font*1.18/72+.08 for row in rows]
    navigation=getattr(pdf,"_publication_navigation",None)
    has_footer=navigation and len(navigation["headings"])>=6
    available=height-.55-title_h-note_h-(.47 if has_footer else .25)
    if sum(heights)+head_h>available and len(df)>1:
        pages=max(2,int((sum(heights)+available-head_h-1e-8)//max(.1,available-head_h)))
        size=max(1,(len(df)+pages-1)//pages)
        for start in range(0,len(df),size):
            _render_pdf_table_page(pdf,title+f'\nRows {start+1}–{min(start+size,len(df))} of {len(df)}',df.iloc[start:start+size],notes=notes,styles=styles)
        return
    total=sum(heights)+head_h;top=height-.30-title_h;bottom=top-total
    assert bottom-note_h>.12,(title,bottom,note_h)
    fig=plt.figure(figsize=(width,height));fig.patch.set_facecolor('white')
    fig.text(.325/width,1-.20/height,title_lines,fontsize=11,weight='bold',va='top')
    ax=fig.add_axes([.325/width,bottom/height,available_width/width,total/height]);ax.axis('off')
    t=ax.table(cellText=rows,colLabels=headers,colWidths=fractions,cellLoc='right',bbox=[0,0,1,1]);t.auto_set_font_size(False);t.set_fontsize(font)
    numeric=numeric_display_columns(df)
    for (r,c),cell in t.get_celld().items():
        cell.set_height(([head_h]+heights)[r]/total);cell.set_linewidth(0);cell.PAD=.035
        cell.set_facecolor(color('table_header') if r==0 else 'white')
        cell_style=(styles or {}).get(df.index[r-1],{}).get(df.columns[c],{}) if r else {}
        cell.set_text_props(ha='right' if df.columns[c] in numeric else 'left',weight='bold' if r==0 or cell_style.get('bold') else 'normal')
    y=1
    for r,h in enumerate([head_h]+heights):
        ax.plot([0,1],[y,y],transform=ax.transAxes,color=color('reference') if r<2 else color('table_rule'),lw=.6 if r<2 else .25,clip_on=False);y-=h/total
    ax.plot([0,1],[0,0],transform=ax.transAxes,color=color('reference'),lw=.6,clip_on=False)
    if notes:fig.text(.325/width,(bottom-.14)/height,'\n'.join(note_lines),fontsize=8,va='top',linespacing=1.2)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for cell in t.get_celld().values():
        a=cell.get_text().get_window_extent(renderer);b=cell.get_window_extent(renderer)
        assert b.x0-.2<=a.x0 and a.x1<=b.x1+.2 and b.y0-.2<=a.y0 and a.y1<=b.y1+.2,(title,cell.get_text().get_text())
    if has_footer:
        for link in table_footer(fig,navigation['headings'],navigation['current']):
            navigation['links'].append(dict(link,page=pdf.get_pagecount()+1))
    pdf.savefig(fig);plt.close(fig)
