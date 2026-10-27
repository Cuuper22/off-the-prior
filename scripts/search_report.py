"""Build comparison.csv, map_elites_cells.csv, input_space_summary.csv, provenance.json and README.md
from the run folders and the transfer outputs in results/search.

  python scripts/search_report.py --dir results/search
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from otp.io import read_json, write_json  # noqa: E402
from otp.search.objective import FAMILIES, GRID_SHAPE, N_ENTROPY_BINS, cell_of  # noqa: E402

THRESHOLDS = (1.5, 2.0, 3.0)
METHODS = ("cmaes", "map_elites", "gradient")
LABEL = {"cmaes": "CMA-ES", "map_elites": "MAP-Elites", "gradient": "gradient (a)"}


def md(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def first_reach(h: pd.DataFrame, thr: float, col: str):
    x = h[(h.feasible) & (h.fitness >= thr)]
    return float(x[col].iloc[0]) if len(x) else float("nan")


def run_stats(h: pd.DataFrame, method: str, seed: int) -> dict:
    h = h.copy()
    h["feasible"] = h["feasible"].astype(str) == "True"
    f = h[h.feasible].sort_values("fitness", ascending=False)
    top = f.head(20)
    r = {"method": method, "seed": seed, "evaluations": len(h), "calls": int(h.calls.max()) if len(h) else 0,
         "n_feasible": int(h.feasible.sum()), "best_fitness": float(f.fitness.max()) if len(f) else 0.0,
         "n_top": len(top), "median_top20": float(top.fitness.median()) if len(top) else float("nan")}
    for t in THRESHOLDS:
        r[f"calls_to_{t}"] = first_reach(h, t, "calls")
    for t in THRESHOLDS:
        r[f"evals_to_{t}"] = first_reach(h, t, "eval")
    cells = {}
    for _, x in f.iterrows():
        c = cell_of(x.family, x.spec_entropy)
        cells[c] = max(cells.get(c, 0.0), float(x.fitness))
    r["cells_filled"] = len(cells)
    for t in THRESHOLDS:
        r[f"cells_ge_{t}"] = sum(v >= t for v in cells.values())
    for t in THRESHOLDS:
        r[f"signatures_ge_{t}"] = int(f[f.fitness >= t].signature.nunique())
    r["families_ge_1.5"] = int(f[f.fitness >= 1.5].family.nunique())
    return r, cells


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/search")
    a = ap.parse_args(argv)
    d = a.dir
    rows, cell_rows = [], []
    for method in METHODS:
        for p in sorted(glob.glob(os.path.join(d, "runs", f"{method}_seed*", "history.csv"))):
            seed = int(p.split("seed")[-1].split(os.sep)[0])
            h = pd.read_csv(p)
            r, cells = run_stats(h, method, seed)
            rows.append(r)
            for (fi, eb), v in sorted(cells.items()):
                cell_rows.append({"method": method, "seed": seed, "family": FAMILIES[fi], "entropy_bin": eb, "best_fitness": v})
    per = pd.DataFrame(rows)
    agg = []
    for method in METHODS:
        x = per[per.method == method]
        if not len(x):
            continue
        r = {"method": method, "seed": "mean"}
        for c in per.columns[2:]:
            vals = x[c].astype(float)
            r[c] = float(vals.mean()) if not c.startswith(("calls_to", "evals_to")) else float(vals.mean(skipna=True))
        for t in THRESHOLDS:
            r[f"n_runs_reaching_{t}"] = int(x[f"calls_to_{t}"].notna().sum())
        agg.append(r)
    comp = pd.concat([per, pd.DataFrame(agg)], ignore_index=True)
    comp.to_csv(os.path.join(d, "comparison.csv"), index=False)
    pd.DataFrame(cell_rows).to_csv(os.path.join(d, "map_elites_cells.csv"), index=False)  # all methods' binned archives

    # input-space summary
    ip = os.path.join(d, "input_space.csv")
    isum = None
    if os.path.exists(ip):
        x = pd.read_csv(ip)
        g = []
        for size, y in x.groupby("size", sort=False):
            yr = y[y.reached.astype(str) == "True"]
            g.append({"size": size, "n_series": len(y), "n_reached": len(yr),
                      "median_rho": float(yr.rho.median()) if len(yr) else float("nan"),
                      "min_rho": float(yr.rho.min()) if len(yr) else float("nan"),
                      "max_rho": float(yr.rho.max()) if len(yr) else float("nan"),
                      "median_l2_over_std": float(yr.l2_over_std.median()) if len(yr) else float("nan"),
                      "median_mase_clean": float(y.mase_clean.median())})
        isum = pd.DataFrame(g)
        isum.to_csv(os.path.join(d, "input_space_summary.csv"), index=False)

    # provenance: merge the invocation records and per-run wall times
    prov = {"invocations": [], "runs": []}
    for p in sorted(glob.glob(os.path.join(d, "provenance_*.json"))) + sorted(glob.glob(os.path.join(d, "*provenance.json"))):
        if os.path.basename(p) == "provenance.json":
            continue
        prov["invocations"] += read_json(p).get("invocations", [])
    for p in sorted(glob.glob(os.path.join(d, "runs", "*", "run.json"))):
        prov["runs"].append(read_json(p))
    tj = os.path.join(d, "transfer.json")
    if os.path.exists(tj):
        prov["transfer"] = read_json(tj)
    write_json(os.path.join(d, "provenance.json"), prov)

    # ---- README
    L = []
    L.append("# results/search: learned failure generator, CMA-ES vs MAP-Elites vs gradient search\n")
    L.append("Facts only. Every number below is copied from the CSV/JSON files in this folder by `scripts/search_report.py`.\n")
    L.append("## Files\n")
    L.append("- `runs/<method>_seed<k>/history.csv`: one row per fresh genome evaluation (evaluation index, cumulative Timestral "
             "calls, best-so-far, the genome, fitness, feasibility, MASE and sCRPS ratios, cov80, oracle/seasonal-naive sCRPS "
             "ratio, descriptors). `elites.json`: top 20 feasible genomes with `describe()` recipes (MAP-Elites also its archive "
             "cells). `run.json`: calls, evaluations, wall time. Gradient runs also have `grad_trace.csv` (one row per step and start).")
    L.append("- `comparison.csv`: per method and search seed, plus a `mean` row per method.")
    L.append("- `map_elites_cells.csv`: for every run of every method, the best fitness per cell when all its feasible genomes "
             "are binned into the MAP-Elites grid.")
    L.append("- `transfer.csv`, `transfer_summary.csv`, `transfer_spearman.csv`, `transfer.json`, `elites.npz`: transfer to other sizes.")
    L.append("- `input_space.csv`, `input_space_summary.csv`: the input-space sensitivity study (b).")
    L.append("- `provenance.json`: command lines, git commits, hosts, wall times.\n")
    L.append("## Terms\n")
    L.append("- Genome: 25 numbers in [0, 1] that decode (`otp/generators/mixture.py`) into a mixture of four component slots "
             "(trend, season, AR noise, chaos, regime and heavy-tail pieces) plus an observation-noise level.")
    L.append("- MASE ratio: MASE of Timestral divided by MASE of the true-dynamics oracle (Monte Carlo continuation of the true "
             "process from its true state, 500 paths), on the same sample, context 2048, horizon 128. Fitness is the geometric "
             "mean of this ratio over 3 fixed seeds (`SEARCH_SEEDS`).")
    L.append("- Feasible: oracle sCRPS is at most 0.8 times seasonal-naive sCRPS (geometric mean over the 3 seeds), everything finite, "
             "context std within [1e-3, 1e6], oracle MASE at least 1e-6. Infeasible genomes have fitness 0; for the optimizers' ranking "
             "they carry a search score of minus the violation (see `otp/search/objective.py`).")
    L.append("- Call: one Timestral forecast of one series. A genome evaluation is 3 calls (3 seeds). A gradient pass of one series "
             "(forward and backward) is 2 calls. Symmetric averaging inside the forecaster is part of one call. Repeated genomes in a "
             "run cost nothing.")
    L.append("- Grid: 4 dominant families (primitive, chaos, regime, tails; family with the largest summed slot weight) x 10 bins of "
             "spectral entropy of the context (mean over seeds, equal bins on [0, 1]) = 40 cells.\n")
    if len(per):
        L.append("## Comparison (comparison.csv, `mean` rows)\n")
        m = comp[comp.seed.astype(str) == "mean"]
        cols = ["method", "evaluations", "calls", "n_feasible", "best_fitness", "median_top20", "calls_to_1.5", "calls_to_2.0",
                "calls_to_3.0", "n_runs_reaching_1.5", "n_runs_reaching_2.0", "n_runs_reaching_3.0"]
        L.append(md(m[cols]) + "\n")
        L.append("`calls_to_x` is the mean over the runs that reached fitness x (the `n_runs_reaching_x` columns count them out of "
                 "the runs of that method).\n")
        cols = ["method", "cells_filled", "cells_ge_1.5", "cells_ge_2.0", "cells_ge_3.0", "signatures_ge_1.5", "signatures_ge_2.0",
                "signatures_ge_3.0", "families_ge_1.5"]
        L.append("### Coverage of the same grid (mean over seeds)\n")
        L.append(md(m[cols]) + "\n")
        L.append("`signatures_ge_x` is the number of distinct sets of components (weight above 0.05) among feasible genomes with "
                 "fitness at least x.\n")
        L.append("### Per search seed\n")
        cols = ["method", "seed", "evaluations", "calls", "best_fitness", "median_top20", "calls_to_1.5", "calls_to_2.0", "calls_to_3.0",
                "cells_filled", "cells_ge_1.5", "cells_ge_2.0", "cells_ge_3.0"]
        L.append(md(comp[comp.seed.astype(str) != "mean"][cols]) + "\n")
    # MAP-Elites archives
    arch = sorted(glob.glob(os.path.join(d, "runs", "map_elites_seed*", "elites.json")))
    if arch:
        L.append("## MAP-Elites archives: elite per cell\n")
        for p in arch:
            e = read_json(p)
            L.append(f"### seed {e['seed']}: {len(e['archive'])} cells filled\n")
            rows_ = []
            for c in e["archive"]:
                rec = c["recipe"]
                parts = [f"{s['component']}{'*' if s['mode'] == 'modulate' else ''}({s['weight']:.2f})" for s in rec["slots"] if s["weight"] > 0.05]
                rows_.append({"family": FAMILIES[c["family_index"]], "entropy_bin": c["entropy_bin"], "fitness": c["fitness"],
                              "recipe (* = modulate, weight)": " ".join(parts), "noise_sd": rec["noise_sd"]})
            L.append(md(pd.DataFrame(rows_)) + "\n")
    # gradient
    gt = sorted(glob.glob(os.path.join(d, "runs", "gradient_seed*", "run.json")))
    if gt:
        L.append("## Gradient search (a): run facts\n")
        rr = []
        for p in gt:
            r = read_json(p)
            tr = pd.read_csv(os.path.join(os.path.dirname(p), "grad_trace.csv"))
            rr.append({"seed": r["seed"], "starts": r["info"]["starts"], "screened_draws": r["info"]["screened"], "steps": r["info"]["n_steps"],
                       "rejected_steps_per_start": str(r["info"]["rejected_steps"]), "calls": r["calls"],
                       "scored_genomes": r["evaluations"], "best_fitness": r["best_fitness"]})
        L.append(md(pd.DataFrame(rr)) + "\n")
    # transfer
    ts = os.path.join(d, "transfer_summary.csv")
    if os.path.exists(ts):
        L.append("## Transfer to other sizes (fresh seeds)\n")
        L.append(md(pd.read_csv(ts)) + "\n")
        L.append("`frac_gt_x` is the share of the method's top genomes whose fitness on that size, recomputed on fresh seeds, exceeds x "
                 "(a genome infeasible on the fresh seeds counts as below). `median_fitness_search_5M` is the same genomes' fitness on "
                 "the search seeds.\n")
        L.append("### Spearman correlation of the MASE ratio between sizes\n")
        L.append(md(pd.read_csv(os.path.join(d, "transfer_spearman.csv"))) + "\n")
    if isum is not None:
        L.append("## Input-space sensitivity (b), not a generator failure\n")
        L.append("For in-prior series (`in_prior(seed)`, seeds 0..n-1), the smallest relative perturbation found by projected gradient "
                 "ascent that doubles Timestral's MASE on the fixed realized future. rho is the per-step RMS of the perturbation "
                 "divided by the context std; `l2_over_std` is the perturbation's L2 norm divided by the context std.\n")
        L.append(md(isum) + "\n")
    L.append("## Exact commands\n")
    L.append("```")
    for inv in prov["invocations"]:
        L.append(f"{inv.get('script')} {' '.join(inv.get('argv', []))}   # git {inv.get('git')}, host {inv.get('host')}, "
                 f"wall {inv.get('wall_seconds', float('nan')):.0f}s")
    L.append("```\n")
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
