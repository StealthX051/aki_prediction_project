"""Verify exported files using Python's standard library; no source VM needed."""
from pathlib import Path
import csv,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]

def main():
 errors=[];count=0
 with (ROOT/'SHA256SUMS.csv').open(newline='',encoding='utf-8') as stream:
  for row in csv.DictReader(stream):
   p=ROOT/row['relative_path'];count+=1
   if not p.is_file():errors.append('Missing: '+row['relative_path']);continue
   h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
   if h.hexdigest()!=row['sha256']:errors.append('Changed: '+row['relative_path'])
 print(f'Checked {count} files; {len(errors)} mismatches.')
 for x in errors[:20]:print(x)
 return bool(errors)
if __name__=='__main__':sys.exit(main())
