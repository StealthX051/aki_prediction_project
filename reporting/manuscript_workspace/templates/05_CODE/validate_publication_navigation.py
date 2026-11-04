"""Verify native PDF/Word navigation against actual exported destinations."""
from pathlib import Path
import json,re,subprocess,zipfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
REL='http://schemas.openxmlformats.org/package/2006/relationships'


def verify_word(path):
    with zipfile.ZipFile(path) as z:
        parts=[name for name in z.namelist() if name.startswith('word/') and name.endswith('.xml') and not '/_rels/' in name]
        roots=[ET.fromstring(z.read(name)) for name in parts]
        names=[e.get('{'+W+'}name') for root in roots for e in root.iter('{'+W+'}bookmarkStart')]
        assert len(names)==len(set(names)),(path,'Duplicate bookmark names')
        ids=[e.get('{'+W+'}id') for root in roots for e in root.iter('{'+W+'}bookmarkStart')]
        assert len(ids)==len(set(ids)),(path,'Duplicate bookmark IDs')
        internal=[e.get('{'+W+'}anchor') for root in roots for e in root.iter('{'+W+'}hyperlink') if e.get('{'+W+'}anchor')]
        assert all(target in names for target in internal),(path,'Broken internal link')
        external=0
        for name in z.namelist():
            if not name.startswith('word/_rels/') or not name.endswith('.rels'):continue
            root=ET.fromstring(z.read(name))
            for rel in root:
                if rel.get('Type','').endswith('/hyperlink') and rel.get('TargetMode')=='External':
                    target=rel.get('Target')
                    if not re.match(r'^[a-zA-Z]+:',target):assert (path.parent/target).is_file(),(path,target)
                    external+=1
        return {'file':str(path.relative_to(ROOT)),'bookmarks':len(names),'internal_links':len(internal),'file_links':external}


def verify_pdf(path,record):
    output=subprocess.check_output(['pdfinfo','-dests',str(path)],text=True)
    destinations={name:int(page) for page,name in re.findall(r'^\s*(\d+)\s+\[.*?\]\s+"(PublicationSection_\d+)"',output,re.M)}
    for i,section in enumerate(record['sections']):assert destinations.get(f'PublicationSection_{i}')==section['page'],(path,i,section,destinations)
    raw=path.read_bytes()
    objects={int(n):body for n,body in re.findall(rb'(\d+) 0 obj\s*(.*?)\s*endobj',raw,re.S)}
    catalog=next(body for body in objects.values() if re.search(rb'/Type\s*/Catalog\b',body))
    pages_root=int(re.search(rb'/Pages\s+(\d+) 0 R',catalog).group(1))
    ordered=[]
    def descend(number):
        body=objects[number]
        if re.search(rb'/Type\s*/Pages\b',body):
            kids=re.search(rb'/Kids\s*\[(.*?)\]',body,re.S).group(1)
            for child in re.findall(rb'(\d+) 0 R',kids):descend(int(child))
        else:ordered.append(number)
    descend(pages_root)
    links=0;pages_with_links=[]
    for page_number,number in enumerate(ordered,1):
        body=objects[number];annots=re.search(rb'/Annots\s*\[(.*?)\]',body,re.S)
        if not annots:continue
        page_links=0
        for annotation in re.findall(rb'(\d+) 0 R',annots.group(1)):
            obj=objects[int(annotation)]
            if not re.search(rb'/Subtype\s*/Link\b',obj):continue
            dest=re.search(rb'/Dest\s*\((PublicationSection_\d+)\)',obj)
            if not dest:continue
            assert dest.group(1).decode() in destinations,(path,'Unresolved link')
            rect=list(map(float,re.search(rb'/Rect\s*\[([^]]+)\]',obj).group(1).split()))
            assert rect[0]>=0 and rect[1]>=0 and rect[2]>rect[0] and rect[3]>rect[1],(path,'Bad link rectangle',rect)
            page_links+=1
        links+=page_links
        if page_links:pages_with_links.append(page_number)
    assert len(ordered)==record['pages'],(path,'Page count')
    assert links==record['internal_links'],(path,'Annotation count',links,record['internal_links'])
    if links:assert pages_with_links==list(range(1,len(ordered)+1)),(path,'Missing footer links',pages_with_links)
    assert b'/Outlines' in raw,(path,'Missing PDF outline')
    return {'file':str(path.relative_to(ROOT)),'pages':len(ordered),'destinations':len(destinations),'internal_links':links,'pages_with_links':len(pages_with_links),'outlines_verified':True}


def main():
    pdf_records=[]
    for sidecar in sorted(ROOT.rglob('*.navigation.json')):
        if any(part in {'project_source','history'} for part in sidecar.parts):continue
        record=json.loads(sidecar.read_text());pdf=sidecar.with_name(sidecar.name.removesuffix('.navigation.json')+'.pdf')
        if pdf.exists():pdf_records.append(verify_pdf(pdf,record))
    word_records=[]
    for folder in ['02_TABLES','01_MANUSCRIPT/publication_tables']:
        for path in sorted((ROOT/folder).rglob('*.docx')):word_records.append(verify_word(path))
    for name in ['Main_Figures_and_Tables','Supplementary_Figures_and_Methods','Supplementary_Tables_Index']:
        word_records.append(verify_word(ROOT/'01_MANUSCRIPT'/(name+'.docx')))
    lookup={r['file']:r for r in word_records}
    assert lookup['01_MANUSCRIPT/Main_Figures_and_Tables.docx']['internal_links']==7
    assert lookup['01_MANUSCRIPT/Supplementary_Figures_and_Methods.docx']['internal_links']==11
    for stem in ['Table_S2_main_all_configurations','Table_S2B_main_paired_differences','Table_S3_broader_ICU_all_configurations','Table_S3B_broader_ICU_paired_differences','Table_S4_family_comparisons']:
        for folder in ['02_TABLES/supplement','01_MANUSCRIPT/publication_tables']:
            entry=lookup[folder+'/'+stem+'.docx'];assert entry['internal_links']>0,(stem,'Missing Word table links')
    report={'status':'passed','pdf_files':pdf_records,'word_files':word_records,'new_navigation_cover_pages':0,'pdf_destination_pages_independently_read_with_poppler':True,'actual_pdf_annotation_targets_verified':True,'actual_word_bookmark_targets_verified':True}
    (ROOT/'00_START_HERE/qa/publication_navigation_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Navigation verified:',len(pdf_records),'PDFs;',len(word_records),'Word files;',sum(r['internal_links'] for r in pdf_records),'native PDF links.')
if __name__=='__main__':main()
