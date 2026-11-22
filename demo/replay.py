"""Run the scientific programs in a temporary workspace and print the measured evidence."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(directory, script, *args):
    print("RUN  " + " ".join([script, *args]), flush=True)
    completed = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", script, *args],
        cwd=directory, check=True, capture_output=True, encoding="utf-8",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return completed.stdout

def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def main():
    with tempfile.TemporaryDirectory(prefix="mira-replay-") as scratch:
        work = Path(scratch) / "附件"
        shutil.copytree(ROOT / "附件", work)
        review = work / "复核实验"
        run(review, "lock_training.py")
        digest = (review / "predictions.sha256").read_text().strip()
        print("训练已锁定 | SHA-256", digest[:16], flush=True)
        run(review, "reveal.py")
        result = read(review / "results.json")
        assert result["prediction_sha256"] == digest
        for group in ("flat", "linear", "curved", "noisy", "heldout_glare"):
            v1 = result["summary"][group]["v1"]; v2 = result["summary"][group]["v2"]
            print(f"{group} | 正确通过 {v1['correct_pass']}/{v1['n']} -> {v2['correct_pass']}/{v2['n']} | v2 拒绝 {v2['rejected']}")
        lin = result["summary"]["linear"]
        print(f"线性组留出 RMSE 中位 | {lin['v1']['median_rmse']} -> {lin['v2']['median_rmse']}")


if __name__ == "__main__":
    main()
