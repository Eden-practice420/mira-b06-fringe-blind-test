"""Evaluate frozen off-grid optical designs. Print auditable JSON, standard library only."""
import argparse
import csv
import json
import math
import random
import statistics
import tempfile
from pathlib import Path

from run import fit


GEOMETRIES = [
    ("single", 83.3, 0.0),
    ("single", 132.7, 0.0),
    ("double", 95.1, 263.4),
    ("double", 123.7, 373.6),
]
REGIMES = {"low_constant": (0.012, 0.), "high_constant": (0.055, 0.),
           "low_sloped": (0.012, 0.055)}


def make_scan(path, width_um, spacing_um, seed, noise, intercept, slope):
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["screen_mm", "intensity_au", "set"])
        for i in range(181):
            x = (i - 90) / 10
            phase = math.pi * x / .6328
            arg = phase * width_um / 1000
            envelope = (math.sin(arg)/arg)**2 if arg else 1.
            profile = envelope * math.cos(phase*spacing_um/1000)**2
            intensity = intercept + slope*x + 1.15*profile + noise*rng.gauss(0.,1.)
            split = "test" if 23 <= i <= 40 or 137 <= i <= 154 else "train"
            writer.writerow([x, round(intensity, 7), split])


def experiment(directory):
    predictions = {}
    truth = {}
    for index, (kind, width, spacing) in enumerate(GEOMETRIES):
        for replicate, seed_offset in enumerate((461,923,1877)):
            seed = seed_offset + 10000*index
            for regime, (noise, slope_size) in REGIMES.items():
                scan_id = f"scan_{len(truth)+1:03d}"
                scan = directory / (scan_id + ".csv")
                slope = slope_size if (index+replicate) % 2 == 0 else -slope_size
                intercept = (0.68, 0.74, 0.80)[replicate]
                make_scan(scan, width, spacing, seed, noise, intercept, slope)
                truth[scan_id] = dict(regime=regime, seed=seed, kind=kind, width_um=width,
                                      spacing_um=spacing, sigma_au=noise, background_au=intercept,
                                      background_slope_au_per_mm=slope)
                predictions[scan_id] = fit(scan)
    records = []
    for scan_id, expected in truth.items():
        pred = predictions[scan_id]
        choice = pred["selected"]
        est = pred["fits"][choice]
        width_error = abs(est["width_um"]-expected["width_um"])
        spacing_error = abs(est["spacing_um"]-expected["spacing_um"])
        correct = (choice == expected["kind"] and width_error <= 2
                   and (expected["kind"] == "single" or spacing_error <= 5))
        assert pred["training"] == 145 and pred["heldout"] == 36
        assert pred["reportable"] == (est["heldout_rmse"] <= .04)
        records.append(dict(id=scan_id,truth=expected,fit=pred,
                            width_error_um=round(width_error, 6),
                            spacing_error_um=round(spacing_error, 6),geometry_correct=correct))
    overview = {}
    for regime in REGIMES:
        rows = [r for r in records if r["truth"]["regime"] == regime]
        accepted = [r for r in rows if r["fit"]["reportable"]]
        correct = sum(r["geometry_correct"] for r in accepted)
        rmses = [r["fit"]["fits"][r["fit"]["selected"]]["heldout_rmse"] for r in rows]
        overview[regime] = dict(n=len(rows),accepted=len(accepted),rejected=len(rows)-len(accepted),
                                correct_delivered=correct,wrong_accepted=len(accepted)-correct,
                                no_correct_delivery_fraction=(len(rows)-correct)/len(rows),
                                misspecified_accepted=len(accepted) if regime=="low_sloped" else None,
                                median_heldout_rmse=statistics.median(rmses))
    return dict(protocol_version=1,summary=overview,cases=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True,help="destination JSON audit file")
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as staging:
        result=experiment(Path(staging))
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result["summary"],indent=2))
