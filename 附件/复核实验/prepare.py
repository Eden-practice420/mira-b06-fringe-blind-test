"""Regenerate shuffle-coded synthetic observations and seal the held-out instrument readings."""
from pathlib import Path
import csv,json,math,random
ROOT=Path(__file__).resolve().parent
train_dir=ROOT/"training";train_dir.mkdir(exist_ok=True)
vault=ROOT/"sealed";vault.mkdir(exist_ok=True)
geometries=[(87.2,0),(133.1,0),(92.6,274.1),(126.4,367.4)]
conditions=("flat","linear","curved","noisy","heldout_glare")
cases=[(w,d,rep,reg) for w,d in geometries for rep in range(2) for reg in conditions]
random.Random(78391).shuffle(cases)
truth={}
with (vault/"heldout.csv").open("w",encoding="utf-8",newline="") as out:
    hw=csv.writer(out);hw.writerow(["scan","screen_mm","intensity_au"])
    for i,(width,spacing,rep,regime) in enumerate(cases):
        name=f"trace-{i+1:02}";seed=14590+round(width*10)*3+rep
        rng=random.Random(seed)
        # Paired random draws across conditions at identical geometry/replicate.
        slope=(.038 if rep==0 else -.046) if regime=="linear" else 0.
        sigma=.070 if regime=="noisy" else .012
        truth[name]=dict(width=width,spacing=spacing,seed=seed,condition=regime,sigma=sigma,slope=slope)
        with (train_dir/(name+".csv")).open("w",encoding="utf-8",newline="") as f:
            tw=csv.writer(f);tw.writerow(["screen_mm","intensity_au"])
            for j in range(181):
                x=(j-90)/10;phase=math.pi*x/.6328
                arg=phase*width/1000
                profile=((math.sin(arg)/arg)**2 if arg else 1.)*math.cos(phase*spacing/1000)**2
                hold=35<=j<=52 or 105<=j<=122
                curve=.010*x*x if regime=="curved" else 0.
                glare=.15 if regime=="heldout_glare" and 106<=j<=121 else 0.
                y=.8+1.15*profile+slope*x+curve+glare+rng.gauss(0,sigma)
                if hold:hw.writerow([name,x,round(y,7)])
                else:tw.writerow([x,round(y,7)])
(vault/"truth.json").write_text(json.dumps(truth,indent=2),encoding="utf-8")
