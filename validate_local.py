#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

REQUIRED = ["image","plate_num","plate_type","bbox","quad","is_vehicle","is_synthetic","source","license","conditions"]
TYPES = {"type1","type1a","type1b","other"}
PLATE_RE = re.compile(r"^[ABEKMHOPCTYX][0-9#]{3}[ABEKMHOPCTYX#]{2}[0-9#]{2,3}$")

def validate_dataset(root: Path) -> int:
    errors=[]; warnings=[]
    meta=root/'meta.csv'
    if not meta.exists():
        print('ERROR: meta.csv not found'); return 2
    with meta.open(encoding='utf-8-sig', newline='') as f:
        r=csv.DictReader(f, delimiter=';')
        if r.fieldnames != REQUIRED:
            errors.append(f"meta.csv columns must be exactly: {REQUIRED}; got {r.fieldnames}")
        rows=list(r)
    counts=Counter(); unique={t:set() for t in TYPES}
    for i,row in enumerate(rows, start=2):
        typ=row.get('plate_type','')
        if typ not in TYPES: errors.append(f'line {i}: invalid plate_type {typ!r}')
        counts[typ]+=1
        plate=row.get('plate_num','')
        if typ!='other' and not PLATE_RE.fullmatch(plate):
            errors.append(f'line {i}: invalid plate_num {plate!r}')
        unique.setdefault(typ,set()).add(plate)
        p=root/row.get('image','')
        if not p.exists(): errors.append(f'line {i}: missing image {row.get("image")}')
        try:
            b=[int(v) for v in row.get('bbox','').split(',')]
            if len(b)!=4 or b[2]<=0 or b[3]<=0: raise ValueError
        except Exception: errors.append(f'line {i}: bbox must be x,y,w,h')
        try:
            q=[int(v) for v in row.get('quad','').split(',')]
            if len(q)!=8: raise ValueError
        except Exception: errors.append(f'line {i}: quad must have 8 integers')
        if row.get('is_vehicle') not in {'0','1'}: errors.append(f'line {i}: is_vehicle must be 0/1')
        if row.get('is_synthetic') not in {'0','1'}: errors.append(f'line {i}: is_synthetic must be 0/1')
    real_rows=[r for r in rows if r.get('is_synthetic')=='0']
    rc=Counter(r['plate_type'] for r in real_rows)
    ru={t:{r['plate_num'] for r in real_rows if r['plate_type']==t} for t in TYPES}
    if rc['type1a']<150 or len(ru['type1a'])<50:
        warnings.append(f"real type1a below recommendation: {rc['type1a']} images, {len(ru['type1a'])} unique (recommended 150/50)")
    if rc['type1b']<300 or len(ru['type1b'])<100:
        warnings.append(f"real type1b below recommendation: {rc['type1b']} images, {len(ru['type1b'])} unique (recommended 300/100)")
    if rc['other']<50:
        warnings.append(f"real other below recommendation: {rc['other']} images (recommended 50)")
    syn=sum(1 for r in rows if r.get('is_synthetic')=='1')
    if syn<5000: warnings.append(f"synthetic count {syn}; recommended at least 5000")

    print('Rows:',len(rows)); print('By type:',dict(counts)); print('Synthetic:',syn,'Real:',len(real_rows))
    for w in warnings: print('WARNING:',w)
    for e in errors[:50]: print('ERROR:',e)
    if len(errors)>50: print(f'... and {len(errors)-50} more errors')
    print('STATUS:', 'PASS' if not errors else 'FAIL')
    return 0 if not errors else 2

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',type=Path,default=Path(__file__).resolve().parent/'dataset')
    args=p.parse_args(); raise SystemExit(validate_dataset(args.dataset))
if __name__=='__main__': main()
