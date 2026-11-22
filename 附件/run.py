"""Blind fitting of screen profiles. Reads only the measured columns and split."""
import csv
import json
import math
from pathlib import Path

FOLDER = Path(__file__).resolve().parent / 'data'


def fit(scan):
    with scan.open(encoding='utf-8') as f:
        points = [(float(r['screen_mm']),float(r['intensity_au']),r['set']) for r in csv.DictReader(f)]
    if len(set(x for x,_,_ in points)) != len(points) or not all(
        math.isfinite(x) and math.isfinite(y) and fold in ('train','test') for x,y,fold in points):
        raise ValueError("Invalid observation or duplicate position")
    train = [(x,y) for x,y,fold in points if fold=='train']
    test = [(x,y) for x,y,fold in points if fold=='test']
    if len(train)<30 or len(test)<10:
        raise ValueError("Insufficient data")
    q = [math.pi*x/0.6328 for x,_ in train+test]
    y = [v for _,v in train]
    ym = sum(y)/len(y)
    best = {}
    for width in range(70, 151, 2):
        env = [(math.sin(v*width/1000)/(v*width/1000))**2 if v else 1 for v in q]
        for separation in [0,*range(200,421,5)]:
            kind = 'single' if separation==0 else 'double'
            profile = [e * math.cos(v*separation/1000)**2 for v,e in zip(q,env)]
            xs = profile[:len(y)]
            mean = sum(xs)/len(xs)
            slope = sum((v-mean)*(obs-ym) for v,obs in zip(xs,y))/sum((v-mean)**2 for v in xs)
            if slope<=0:
                continue
            intercept = ym-slope*mean
            prediction = [intercept+slope*v for v in profile]
            err = sum((v-obs)**2 for v,obs in zip(prediction,y))
            if kind not in best or err<best[kind]['sse']:
                best[kind] = dict(width_um=width,spacing_um=separation,gain=slope,background=intercept,sse=err,
                     heldout_rmse=math.sqrt(sum((a-b)**2 for a,(_,b) in zip(prediction[len(y):],test))/len(test)))
    chosen = min(best, key=lambda model:best[model]['sse'])
    accepted = best[chosen]['heldout_rmse'] <= .04
    return dict(id=scan.stem,training=len(train),heldout=len(test),fits=best,selected=chosen,
                reportable=accepted,reason='within noise budget' if accepted else 'held-out residual exceeds 0.04 AU')


if __name__=='__main__':
    print(json.dumps([fit(p) for p in sorted(FOLDER.glob('scan_*.csv'))],indent=2))
