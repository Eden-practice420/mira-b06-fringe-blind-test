"""This entry point accepts ONLY training CSVs. It never reads the heldout vault."""
import argparse,csv,hashlib,json,math
from pathlib import Path
from statistics import mean
from functools import lru_cache

def profile(x,w,d):
    phase=math.pi*x/.6328
    a=phase*w/1000
    return ((math.sin(a)/a)**2 if a else 1.)*math.cos(phase*d/1000)**2

@lru_cache(None)
def basis(xs):
    n=len(xs);xm=mean(xs);xc=[x-xm for x in xs];xx=sum(x*x for x in xc)
    rows=[]
    for w in range(70,151,2):
        for d in (0,*range(200,421,5)):
            values=[profile(x,w,d) for x in xs]
            pm=mean(values);p=[v-pm for v in values];pp=sum(v*v for v in p);xp=sum(a*b for a,b in zip(xc,p))
            rows.append((w,d,pm,p,pp,xp))
    return xm,xc,xx,rows

def fit(path):
    with path.open(encoding="utf-8") as f:
        rows=[(float(r["screen_mm"]),float(r["intensity_au"])) for r in csv.DictReader(f)]
    rows.sort()
    if len(rows)<30 or len({r[0] for r in rows})!=len(rows) or any(not math.isfinite(v) for r in rows for v in r):raise ValueError("invalid training scan")
    xs=tuple(x for x,y in rows);ys=[y for x,y in rows];n=len(ys)
    xm,xc,xx,grid=basis(xs);ym=mean(ys);yc=[y-ym for y in ys];yy=sum(y*y for y in yc);xy=sum(x*y for x,y in zip(xc,yc))
    best={}
    for w,d,pm,pc,pp,xp in grid:
        py=sum(p*y for p,y in zip(pc,yc))
        for linear in (False,True):
            residual_var=pp-xp*xp/xx if linear else pp
            if residual_var<1e-12:continue
            cov=py-xp*xy/xx if linear else py
            gain=cov/residual_var
            if gain<=0:continue
            slope=(xy-gain*xp)/xx if linear else 0.
            intercept=ym-gain*pm-slope*xm
            sse=max(1e-18,yy-gain*py-slope*xy)
            # Relative information criterion penalizes only the added background slope.
            criterion=n*math.log(sse/n)+int(linear)*math.log(n)
            key="linear" if linear else "constant"
            if key not in best or sse<best[key]["sse"]:
                best[key]=dict(width=w,spacing=d,gain=gain,intercept=intercept,slope=slope,sse=sse,criterion=criterion)
    if len(best)!=2:raise ValueError("no positive-gain candidate")
    return {"points":n,"candidates":best,"v1":"constant","v2":min(best,key=lambda k:best[k]["criterion"]),
            "unpenalized":min(best,key=lambda k:best[k]["sse"])}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("training",type=Path);p.add_argument("out",type=Path);args=p.parse_args()
    fits={f.stem:fit(f) for f in sorted(args.training.glob("*.csv"))}
    if not fits:raise ValueError("missing training data")
    args.out.write_text(json.dumps(fits,indent=2),encoding="utf-8")
    digest=hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(".sha256").write_text(digest+"\n",encoding="ascii");print(digest)
