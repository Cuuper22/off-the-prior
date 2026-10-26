"""Write the Colab notebooks under notebooks/ (kept as code so they stay reviewable and diffable).

python scripts/make_notebooks.py
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


GET_REPO = r'''
# Option A: clone the repo (fill in the URL once it is on GitHub)
REPO_URL = ""  # e.g. "https://github.com/<user>/off-the-prior.git"
import os, subprocess, zipfile
if REPO_URL and not os.path.exists("off-the-prior"):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "off-the-prior"], check=True)
# Option B: upload off-the-prior.zip (made with `git archive --format=zip -o off-the-prior.zip HEAD`)
if not os.path.exists("off-the-prior") and os.path.exists("off-the-prior.zip"):
    with zipfile.ZipFile("off-the-prior.zip") as z:
        z.extractall("off-the-prior")
assert os.path.exists("off-the-prior"), "set REPO_URL or upload off-the-prior.zip to the Colab file pane"
%cd off-the-prior
'''

INSTALL = r'''
!pip -q install "git+https://github.com/mistralai/timestral.git" statsmodels cma
!pip -q install -e . --no-deps
import torch, os
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
os.environ["OTP_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["OTP_THREADS"] = str(os.cpu_count())
'''


def sweep_notebook() -> dict:
    return notebook([
        md("""
# Off the Prior: big-model failure-map sweep (Colab GPU)

Runs the failure-map sweeps for **all four Timestral sizes** (5M, 28M, 123M, 209M) on every
grid in `otp/grids/`, at contexts up to 32,768 and horizons up to 2,048 (2,048 = a rollout
past one 1,024-step decode block). The CPU VM where the project was developed ran 5M/28M on
the full grids and the big sizes on reduced grids; this notebook fills in the rest.

Results land in `results/map_colab/<grid>/` (scores.csv, provenance.json with the GPU name).
The last cell zips them; download the zip, unzip it at the repo root, and commit.

Runtime: GPU (any; an A100 or L4 is faster). Classical baselines (ETS/ARIMA) run on the CPU
cores; set `CLASSICAL = ["seasonal_naive"]` to skip the slow ones.
"""),
        code(GET_REPO),
        code(INSTALL),
        code(r'''
# Configuration
SIZES = ["5M", "28M", "123M", "209M"]
CONTEXTS = [512, 2048, 8192, 32768]
HORIZONS = [16, 128, 1024, 2048]
SEEDS = 16
CLASSICAL = ["seasonal_naive", "ets", "arima"]
BATCH = 32
import pkgutil, otp.grids
GRIDS = sorted(m.name for m in pkgutil.iter_modules(otp.grids.__path__))
print("grids:", GRIDS)
'''),
        code(r'''
import subprocess, sys, time
for g in GRIDS:
    out = f"results/map_colab/{g}"
    if os.path.exists(f"{out}/scores.csv"):
        print("skip (done)", g); continue
    cmd = [sys.executable, "scripts/sweep.py", "--grid", f"otp.grids.{g}", "--sizes", *SIZES,
           "--contexts", *map(str, CONTEXTS), "--horizons", *map(str, HORIZONS), "--seeds", str(SEEDS),
           "--classical", *CLASSICAL, "--batch-size", str(BATCH), "--out", out]
    print(" ".join(cmd)); t = time.time()
    subprocess.run(cmd, check=True)
    print(g, "done in", round(time.time() - t), "s")
'''),
        code(r'''
# Covariate experiments at all sizes (if the covariate runner is present)
if os.path.exists("scripts/covariate_sweep.py"):
    !python scripts/covariate_sweep.py --help | head -30
    # Fill in the full-grid flags printed above, writing to results/covariates_colab/
'''),
        code(r'''
import shutil
shutil.make_archive("map_colab_results", "zip", ".", "results/map_colab")
print("download map_colab_results.zip from the file pane; unzip at the repo root and commit")
'''),
    ])


def main():
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    path = os.path.join(ROOT, "notebooks", "sweep_colab.ipynb")
    with open(path, "w") as f:
        f.write(json.JSONEncoder(indent=1).encode(sweep_notebook()) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
