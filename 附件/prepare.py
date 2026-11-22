"""Fixed synthetic instrument scans. Never imported by the inference engine."""
import csv
import math
import random
from pathlib import Path

folder = Path(__file__).resolve().parent / "data"
folder.mkdir(exist_ok=True)
for label, aperture_mm, separation_mm, trend in [('scan_a',0.12,0,0), ('scan_b',0.10,0.32,0), ('scan_c',0.10,0.32,0.055)]:
    generator = random.Random(830 + ord(label[-1]))
    with (folder / (label+'.csv')).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['screen_mm','intensity_au','set'])
        for index in range(181):
            x = (index-90)*0.1
            # Conversion: lambda=0.0006328 mm; propagation distance=1000 mm.
            q = math.pi * x / (1000*0.0006328)
            env = (math.sin(q*aperture_mm)/(q*aperture_mm))**2 if q else 1
            y = 0.64 + trend*x + 1.15*env*math.cos(q*separation_mm)**2 + generator.gauss(0,0.012)
            writer.writerow([round(x,3),round(y,7),'test' if 23<=index<=40 or 137<=index<=154 else 'train'])
