"""Move the outputs of a Colab search job (flat folders `<prefix><method>_seed<k>` under a pulled
`results/colab/<job>/results/colab_out/`) into the layout of `results/search/`.

  python scripts/search_collect.py --job results/colab/srch-a2 --prefix srch_ --out results/search

Run folders go to `<out>/runs/<method>_seed<k>/`; the input-space csv/log to `<out>/`; the invocation
records (`*provenance.json`) to `<out>/provenance_<name>.json`.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--prefix", default="srch_")
    ap.add_argument("--out", default="results/search")
    a = ap.parse_args(argv)
    src = os.path.join(a.job, "results", "colab_out")
    for d in sorted(glob.glob(os.path.join(src, a.prefix + "*_seed*"))):
        name = os.path.basename(d)[len(a.prefix):]
        dst = os.path.join(a.out, "runs", name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(d, dst)
        print("run", name)
    for d in sorted(glob.glob(os.path.join(src, "*input_space"))):
        for f in os.listdir(d):
            shutil.copy(os.path.join(d, f), os.path.join(a.out, f))
            print("input-space file", f)
    for f in sorted(glob.glob(os.path.join(src, "*provenance.json"))):
        shutil.copy(f, os.path.join(a.out, "provenance_" + os.path.basename(f).replace("provenance.json", "").rstrip("_") + ".json"))
        print("provenance", f)


if __name__ == "__main__":
    main()
