#!/usr/bin/env python3
from pathlib import Path
import cv2, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.pipeline import ANPRPipeline
expected={'type1.png':('A123BC77','type1'),'type1a.png':('M456OP197','type1a'),'type1b.png':('T777YX99','type1b')}
p=ANPRPipeline(ROOT/'models')
for name,(num,typ) in expected.items():
    out=p.predict(cv2.imread(str(ROOT/'sample_input'/name)))
    assert out and out[0]['plate_num']==num and out[0]['plate_type']==typ,(name,out)
    print(name,'OK',out[0]['plate_num'],out[0]['plate_type'],out[0]['confidence'])
print('SMOKE PASS')
