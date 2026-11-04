"""Portable Word and PDF navigation; no third-party PDF library required."""
from pathlib import Path
import json, re, subprocess, tempfile
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from publication_style import WORD_FONT, color


def bookmark(paragraph, name, number):
    """Mark an existing paragraph without adding visible text or changing geometry."""
    start=OxmlElement('w:bookmarkStart');start.set(qn('w:id'),str(number));start.set(qn('w:name'),name)
    end=OxmlElement('w:bookmarkEnd');end.set(qn('w:id'),str(number))
    paragraph._p.insert(0,start);paragraph._p.append(end)


def hyperlink(paragraph, text, *, anchor=None, target=None, size=9):
    """Add a true Word link; internal targets use bookmarks, external ones relationships."""
    link=OxmlElement('w:hyperlink')
    if anchor:link.set(qn('w:anchor'),anchor)
    else:
        from docx.opc.constants import RELATIONSHIP_TYPE
        link.set(qn('r:id'),paragraph.part.relate_to(target,RELATIONSHIP_TYPE.HYPERLINK,is_external=True))
    run=OxmlElement('w:r');prop=OxmlElement('w:rPr')
    font=OxmlElement('w:rFonts');font.set(qn('w:ascii'),WORD_FONT);font.set(qn('w:hAnsi'),WORD_FONT);prop.append(font)
    sz=OxmlElement('w:sz');sz.set(qn('w:val'),str(int(size*2)));prop.append(sz)
    ink=OxmlElement('w:color');ink.set(qn('w:val'),color('ink').lstrip('#'));prop.append(ink)
    underline=OxmlElement('w:u');underline.set(qn('w:val'),'single');prop.append(underline)
    run.append(prop);t=OxmlElement('w:t');t.text=text;run.append(t);link.append(run);paragraph._p.append(link)
    return link


def short_label(text):
    return (text.replace('AKI within 72 hours','AKI').replace('Logistic regression','LR')
      .replace('Main shared cohort','Main cohort').replace('Discrimination and Brier score','Discrimination')
      .replace('Classification at fold thresholds','Classification'))


def word_section_index(document, headings):
    """Compact grouped links for long native tables, without a separate cover page."""
    if len(headings)<6:return
    groups={}
    for i,heading in enumerate(headings):
        parts=heading.split(' — ');prefix=' — '.join(parts[:-1]);leaf=parts[-1]
        groups.setdefault(prefix,[]).append((i,leaf))
    for prefix,items in groups.items():
        p=document.add_paragraph();p.paragraph_format.space_after=Pt(2)
        if prefix:p.add_run(short_label(prefix)+': ').bold=True
        for n,(i,leaf) in enumerate(items):
            if n:p.add_run(' · ')
            hyperlink(p,short_label(leaf),anchor=f'TableSection_{i}')


def word_book_footer(document, groups):
    """A compact linked display index in the existing footer area."""
    p=document.sections[0].footer.paragraphs[0]
    p.paragraph_format.space_before=p.paragraph_format.space_after=Pt(0)
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for g,(label,links) in enumerate(groups):
        if g:p.add_run('  |  ')
        p.add_run(label+' ')
        for n,(text,anchor) in enumerate(links):
            if n:p.add_run(' · ')
            hyperlink(p,text,anchor=anchor,size=8)
    for r in p.runs:r.font.name=WORD_FONT;r.font.size=Pt(8)


def table_footer(fig, headings, current):
    """Draw linked hierarchy choices; return target indexes and PDF-point rectangles."""
    if len(headings)<6:return []
    current_parts=headings[current].split(' — ');groups=[]
    for depth in range(len(current_parts)):
        siblings={}
        for i,heading in enumerate(headings):
            parts=heading.split(' — ')
            if len(parts)>depth and parts[:depth]==current_parts[:depth]:siblings.setdefault(parts[depth],i)
        if len(siblings)>1:groups.append([(short_label(k),v,k==current_parts[depth]) for k,v in siblings.items()])
    groups.append([('Previous',max(0,current-1),False),('Next',min(len(headings)-1,current+1),False)])
    artists=[];x=24.;width,height=fig.get_size_inches()*72
    renderer=fig.canvas.get_renderer()
    def text(value,weight='normal'):
        nonlocal x
        a=fig.text(x/width,10/height,value,fontsize=8,weight=weight,va='bottom')
        box=a.get_window_extent(renderer);x+=box.width*72/fig.dpi
        return a
    text('Jump to: ')
    for g,group in enumerate(groups):
        if g:text('  |  ')
        for n,(label,index,active) in enumerate(group):
            if n:text(' · ')
            a=text(label,'bold' if active else 'normal');artists.append((a,index))
    assert x<width-20,('Navigation footer exceeds page width',headings[current],x,width)
    links=[]
    for a,index in artists:
        b=a.get_window_extent(renderer);links.append({'target':index,'rect':[v*72/fig.dpi for v in [b.x0-1,b.y0-1,b.x1+1,b.y1+1]]})
    return links


def pdf_pages(path):
    output=subprocess.check_output(['pdfinfo',str(path)],text=True)
    return int(re.search(r'^Pages:\s+(\d+)',output,re.M).group(1))


def _pdf_string(text):
    return '<FEFF'+text.encode('utf-16-be').hex().upper()+'>'


def add_pdf_navigation(path, title, sections, links=()):
    """Add native destinations, hierarchical outline entries and internal links via pdfmark.

    ``sections`` is a sequence of (heading, one-based page). It describes existing
    pages only; Ghostscript preserves vector content and original page dimensions.
    """
    path=Path(path);page_count=pdf_pages(path)
    assert sections and all(1<=page<=page_count for _,page in sections)
    destinations=[f'PublicationSection_{i}' for i in range(len(sections))]
    marks=['%!PS-Adobe-3.0','/pdfmark where {pop} { /pdfmark {cleartomark} bind def } ifelse']
    for name,(_,page) in zip(destinations,sections):
        marks.append(f'[ /Dest /{name} /Page {page} /View [/Fit] /DEST pdfmark')
    tree={}
    for heading,page in sections:
        node=tree
        for part in heading.split(' — '):
            item=node.setdefault(part,{'page':page,'children':{}});node=item['children']
    def outline(nodes):
        for label,node in nodes.items():
            count=f" /Count -{len(node['children'])}" if node['children'] else ''
            marks.append(f"[ /Title {_pdf_string(label)} /Page {node['page']} /View [/Fit]{count} /OUT pdfmark")
            outline(node['children'])
    marks.append(f'[ /Title {_pdf_string(title)} /Page 1 /View [/Fit] /Count {len(tree)} /OUT pdfmark')
    outline(tree)
    # ANN pdfmarks attach to the current page. /Page on ANN is ignored by gs;
    # emit them from EndPage after each actual page, rather than all at startup.
    if links:
        by_page={page:[] for page in range(1,page_count+1)}
        for link in links:
            assert 1<=link['page']<=page_count and 0<=link['target']<len(sections)
            rect=' '.join(f'{v:.3f}' for v in link['rect'])
            by_page[link['page']].append(f"[ /Subtype /Link /Rect [{rect}] /Border [0 0 0] /Dest /{destinations[link['target']]} /ANN pdfmark")
        marks.append('/PublicationLinkMarks [')
        for page in range(1,page_count+1):marks.append('{ '+ ' '.join(by_page[page])+' }')
        marks.append('] def /PublicationPage 0 def')
        marks.append('<< /EndPage { 2 eq { pop false } { pop /PublicationPage PublicationPage 1 add def PublicationPage PublicationLinkMarks length le { PublicationLinkMarks PublicationPage 1 sub get exec } if true } ifelse } >> setpagedevice')
    marks.append('[ /PageMode /UseOutlines /DOCVIEW pdfmark')
    with tempfile.TemporaryDirectory(prefix='publication-navigation-') as temp:
        script=Path(temp)/'navigation.ps';script.write_text('\n'.join(marks)+'\n')
        out=Path(temp)/'navigated.pdf'
        subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=pdfwrite','-dCompatibilityLevel=1.7','-sOutputFile='+str(out),str(script),str(path)],check=True)
        assert pdf_pages(out)==page_count
        content=out.read_bytes()
        assert b'/Outlines' in content and content.count(b'/Subtype/Link')+content.count(b'/Subtype /Link')>=len(links)
        out.replace(path)
    return {'path':path.name,'pages':page_count,'sections':[{'heading':h,'page':p} for h,p in sections],
            'internal_links':len(links),'hierarchical_outlines':True,'new_pages':0}
