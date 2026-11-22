"""Unlock held-out observations only after the predictions have been committed to a digest."""
import csv,hashlib,json,math,sys,tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median
ROOT=Path(__file__).resolve().parent
p=ROOT/"predictions.json"
digest=hashlib.sha256(p.read_bytes()).hexdigest()
if digest!=(ROOT/"predictions.sha256").read_text().strip():raise ValueError("Frozen prediction digest mismatch")
preds=json.loads(p.read_text(encoding="utf-8"))
truth=json.loads((ROOT/"sealed/truth.json").read_text(encoding="utf-8"))
# The blind training view is byte-identical for matched flat/glare conditions.
flat={(v["width"],v["spacing"],v["seed"]):s for s,v in truth.items() if v["condition"]=="flat"}
for name,v in truth.items():
    if v["condition"]=="heldout_glare":
        a=ROOT/"training"/(name+".csv")
        b=ROOT/"training"/(flat[(v["width"],v["spacing"],v["seed"])] + ".csv")
        assert a.read_bytes()==b.read_bytes()
held=defaultdict(list)
with (ROOT/"sealed/heldout.csv").open(encoding="utf-8") as f:
    for r in csv.DictReader(f):held[r["scan"]].append((float(r["screen_mm"]),float(r["intensity_au"])))
if set(held)!=set(truth) or set(held)!=set(preds):raise ValueError("scan keys do not match")
from blind_fit import profile
sys.path.insert(0,str(ROOT.parent))
from run import fit as legacy_fit
records=[];points=[];cross_checked=set()
for stem,fit in preds.items():
    t=truth[stem]
    with (ROOT/"training"/(stem+".csv")).open(encoding="utf-8") as f:
        train=[(float(r["screen_mm"]),float(r["intensity_au"])) for r in csv.DictReader(f)]
    assert len(train)==145 and len(held[stem])==36 and not ({r[0] for r in train}&{r[0] for r in held[stem]})
    for method in ("v1","unpenalized","v2"):
        background=fit[method];m=fit["candidates"][background]
        residuals=[]
        for x,y in held[stem]:
            yhat=m["intercept"]+m["gain"]*profile(x,m["width"],m["spacing"])+m["slope"]*x
            residuals.append(yhat-y);points.append([stem,method,x,y,yhat,yhat-y])
        rmse=math.sqrt(sum(r*r for r in residuals)/len(residuals));ok=rmse<=.04
        correct=abs(m["width"]-t["width"])<=2 and ((m["spacing"]==0)==(t["spacing"]==0)) and abs(m["spacing"]-t["spacing"])<=5
        records.append(dict(id=stem,method=method,condition=t["condition"],background=background,width=m["width"],spacing=m["spacing"],slope=m["slope"],truth_width=t["width"],truth_spacing=t["spacing"],correct=correct,accepted=ok,rmse=rmse))
        if method=="v1" and t["condition"] not in cross_checked:
            # Independently compare our frozen constant-background fit with the original v1 source code.
            with tempfile.TemporaryDirectory() as tmp:
                tmpfile=Path(tmp)/"cross.csv"
                with tmpfile.open("w",newline="",encoding="utf-8") as f:
                    wr=csv.writer(f);wr.writerow(["screen_mm","intensity_au","set"]);wr.writerows([x,y,"train"] for x,y in train);wr.writerows([x,y,"test"] for x,y in held[stem])
                v1=legacy_fit(tmpfile);v=v1["fits"][v1["selected"]]
                assert (v["width_um"],v["spacing_um"])==(m["width"],m["spacing"]) and abs(v["heldout_rmse"]-rmse)<1e-8
            cross_checked.add(t["condition"])
summary={}
for reg in ("flat","linear","curved","noisy","heldout_glare"):
    summary[reg]={}
    for method in ("v1","unpenalized","v2"):
        g=[r for r in records if r["method"]==method and r["condition"]==reg];assert len(g)==8
        summary[reg][method]=dict(n=8,linear=sum(r["background"]=="linear" for r in g),correct_pass=sum(r["accepted"] and r["correct"] for r in g),
                                  wrong_pass=sum(r["accepted"] and not r["correct"] for r in g),rejected=sum(not r["accepted"] for r in g),
                                  median_rmse=round(median(r["rmse"] for r in g),6))
with (ROOT/"heldout_residuals.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.writer(f);w.writerow(["id","method","screen_mm","intensity","prediction","residual"]);w.writerows(points)
(ROOT/"results.json").write_text(json.dumps(dict(prediction_sha256=digest,legacy_cross_checks=sorted(cross_checked),summary=summary,cases=records),indent=2),encoding="utf-8")
print(json.dumps(summary))
