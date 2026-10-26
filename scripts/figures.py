"""Paper figures from the lead summaries (results/map/*/summary.csv made by scripts/summarize.py).

python scripts/figures.py --out paper/figures

Every figure is drawn only from summary files; the figure script writes a JSON sidecar
listing the source files it read.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from otp.io import write_json  # noqa: E402

SIZE_ORDER = ["timestral-5M", "timestral-28M", "timestral-123M", "timestral-209M"]
COLORS = {"timestral-5M": "#2a7fbf", "timestral-28M": "#1b9e77", "timestral-123M": "#d95f02",
          "timestral-209M": "#7570b3"}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_summaries(root: str) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(glob.glob(os.path.join(root, "*", "summary.csv")))
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["source"] = f
        frames.append(d)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df["knob_num"] = df["knob_value"].map(_num)
    return df, files


def control_level(df: pd.DataFrame, forecaster: str, T: int, H: int) -> float:
    c = df[(df.generator == "in_prior") & (df.forecaster == forecaster) & (df["T"] == T) & (df.H == H)]
    return float(np.exp(np.log(c.mase_ratio_oracle_gmean).mean())) if len(c) else np.nan


def failure_curves(df: pd.DataFrame, generators: list[str], T: int, H: int, path: str,
                   metric: str = "mase_ratio_oracle", group_by: str | None = None) -> None:
    n = len(generators)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax, gen in zip(axes.flat, generators):
        g = df[(df.generator == gen) & (df["T"] == T) & (df.H == H)]
        for fc in SIZE_ORDER:
            s = g[g.forecaster == fc]
            if group_by:
                s = s.groupby("knob_num", as_index=False)[[f"{metric}_gmean", f"{metric}_lo95", f"{metric}_hi95"]].median()
            s = s.sort_values("knob_num")
            if not len(s):
                continue
            ax.plot(s.knob_num, s[f"{metric}_gmean"], "-o", ms=3, color=COLORS[fc], label=fc.replace("timestral-", ""))
            ax.fill_between(s.knob_num, s[f"{metric}_lo95"], s[f"{metric}_hi95"], color=COLORS[fc], alpha=0.15)
        lvl = control_level(df, "timestral-28M", T, H)
        if np.isfinite(lvl):
            ax.axhline(lvl, color="gray", ls="--", lw=1, label="in-prior (28M)")
        ax.axhline(1.0, color="black", lw=0.6)
        ax.set_yscale("log")
        ax.set_title(gen, fontsize=10)
        k = g.knob.dropna().iloc[0] if "knob" in g and g.knob.notna().any() else "knob"
        ax.set_xlabel(str(k))
        ax.set_ylabel("MASE / oracle MASE" if metric.startswith("mase") else metric)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"context {T}, horizon {H}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/map")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--H", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, files = load_summaries(a.root)
    if "knob" not in df:
        df["knob"] = None
    made = []
    for fam, gens in df.groupby(df.source.map(lambda p: os.path.basename(os.path.dirname(p)))).generator.unique().items():
        gens = [g for g in gens if g != "in_prior"]
        if not gens:
            continue
        p = os.path.join(a.out, f"curves_{fam}_T{a.T}_H{a.H}.png")
        failure_curves(df, sorted(gens), a.T, a.H, p, group_by="median")
        made.append(p)
    write_json(os.path.join(a.out, "figures_sources.json"), {"inputs": files, "figures": made})
    print("\n".join(made))


if __name__ == "__main__":
    main()
