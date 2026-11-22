"""Stage the training-only run in a fresh directory. Do not copy sealed readings here."""
import shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as t:
    stage=Path(t);shutil.copytree(ROOT/"training",stage/"training");shutil.copy2(ROOT/"blind_fit.py",stage/"blind_fit.py")
    assert sorted(f.name for f in stage.iterdir())==["blind_fit.py","training"]
    subprocess.run([sys.executable,"-B","-X","utf8","blind_fit.py","training","predictions.json"],cwd=stage,check=True)
    for ext in ("json","sha256"):shutil.copyfile(stage/("predictions."+ext),ROOT/("predictions."+ext))
