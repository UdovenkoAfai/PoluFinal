from pathlib import Path
import csv
import cv2

from app.pipeline import ANPRPipeline

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'type1.png': ('A123BC77','type1'),
    'type1a.png': ('M456OP197','type1a'),
    'type1b.png': ('T777YX99','type1b'),
}

def test_sample_inputs():
    p = ANPRPipeline(ROOT/'models')
    for name,(num,typ) in EXPECTED.items():
        im=cv2.imread(str(ROOT/'sample_input'/name))
        out=p.predict(im)
        assert out, name
        assert out[0]['plate_num']==num, (name,out[0])
        assert out[0]['plate_type']==typ, (name,out[0])
